package org.qualet.irl.light.shadow;

import it.unimi.dsi.fastutil.floats.FloatArrayList;
import it.unimi.dsi.fastutil.longs.Long2IntOpenHashMap;
import it.unimi.dsi.fastutil.longs.Long2ObjectMap;
import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.longs.LongSet;
import it.unimi.dsi.fastutil.objects.ObjectIterator;
import org.joml.Matrix4f;
import org.joml.Vector3f;
import org.lwjgl.opengl.GL11;
import org.lwjgl.opengl.GL13;
import org.lwjgl.opengl.GL15;
import org.lwjgl.opengl.GL20;
import org.lwjgl.opengl.GL30;
import org.lwjgl.opengl.GL33;
import org.lwjgl.system.MemoryUtil;

import java.nio.FloatBuffer;
import java.util.List;
import java.util.ArrayList;
import net.minecraft.client.Minecraft;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.client.renderer.block.BlockStateModelSet;
import net.minecraft.client.renderer.block.dispatch.BlockStateModel;
import net.minecraft.client.renderer.block.dispatch.BlockStateModelPart;
import net.minecraft.client.renderer.texture.TextureAtlas;
import com.mojang.renderpearl.backend.opengl.GlTexture;
import com.mojang.renderpearl.backend.opengl.GlSampler;
import net.minecraft.client.renderer.chunk.ChunkSectionLayer;
import net.minecraft.client.resources.model.geometry.BakedQuad;
import net.minecraft.core.BlockPos;
import net.minecraft.util.RandomSource;
import net.minecraft.world.level.block.state.BlockState;

/**
 * Bakes spot/point shadow depth maps into the per-light atlas tiles / cube faces.
 *
 * <p><b>MC 1.21.11 line — occluders rasterized with a self-contained raw-GL depth
 * backend.</b> The 1.21.4 baker drew occluders with MC's {@code VertexBuffer} +
 * {@code RenderSystem} matrices + a vanilla entity {@code Immediate}, all removed
 * by the 1.21.5 RenderPipeline rework and the 1.21.9 EntityRenderState rewrite.
 * Rather than chase MC's {@code GpuDevice}/{@code RenderPass} model, the bake now
 * owns two tiny GLSL depth programs (a POSITION-only {@code gl_Position =
 * uViewProj * pos} program for opaque blocks + entity casters, and a POSITION+UV
 * program that samples the block atlas and discards transparent texels for cutout
 * blocks) plus VAOs and per-light VBO caches, and draws straight into the bound
 * depth FBO. This is independent of MC's render pipeline, so it is robust across
 * MC versions.</p>
 *
 * <p><b>The orchestration, filtering, and caster seam are UNCHANGED from the
 * 1.21.4 line</b> — only the drawing layer swapped. The pass lifecycle
 * ({@link #beginBake}/{@link #beginSpot}/{@link #beginPointFace}/{@link #endPass}),
 * the caster seam ({@link #beginCasterBatch}/{@link #emitCaster}/{@link #endCasterBatch}
 * over {@link ShadowCasterSource}/{@link OccluderSink}/{@link OccluderBatch}), the
 * per-light dirty VBO cache keyed by {@code LightRegistry.id}, and the
 * spot/point/MSM cube-view math all read exactly as before; {@link ShadowBaker}
 * drives them identically.</p>
 *
 * <p><b>Caster geometry.</b> On this line the {@link OccluderBatch} backend is
 * {@link RawOccluderBatch}: a per-mod {@link ShadowCasterSource#emitOccluder}
 * downcasts it and {@code append}s the caster's REAL model silhouette as world-space
 * POSITION triangles (produced by the mod's own capture path — the 1.21.11
 * equivalent of the removed immediate entity draw). The pass accumulates all its
 * casters and flushes them once with a raw {@code glDrawArrays} in {@link #endPass}.
 * Block occluders flow through the separate {@link #renderBlocksDepth} helper.</p>
 *
 * <p>GL-state safety: every global bit each draw touches (program, VAO, array
 * buffer, depth test/func/mask, cull, active texture + its 2D binding) is
 * snapshotted with {@code glGet*} and restored to its captured value, and the
 * FBO/viewport/scissor are restored by {@link #endPass}. MC's {@code GlStateManager}
 * caches the FBO binding but NOT program/VAO/buffer; restoring the actual binding
 * keeps MC's cache consistent.</p>
 */
public final class ShadowRenderer
{
    private static final float NEAR = 0.05f;

    private static boolean inPass = false;
    /** True once {@link #savePassState} has snapshotted the original GL state this
     *  bake. {@link #beginBake} clears it so the next bake re-grabs the (possibly
     *  different) state, while passes within one bake share the single snapshot
     *  instead of re-running the glGet* sync points. */
    private static boolean passStateSaved = false;

    private static int savedFbo;
    private static final int[] savedViewport = new int[4];
    private static boolean savedScissorEnabled;
    private static final int[] savedScissorBox = new int[4];
    private static boolean savedDepthMask;

    /** The current light's view/projection (world space), combined into
     *  {@link #viewProj} for the depth programs' uViewProj uniform. The block +
     *  caster draws upload this directly, NEVER MC's live modelview (which no
     *  longer exists on the hot path here), so a caster can never corrupt them. */
    private static final Matrix4f currentView = new Matrix4f();
    private static final Matrix4f currentProj = new Matrix4f();
    private static final Matrix4f viewProj = new Matrix4f();

    /** Per-pass light-relative bake anchor A = round((float) lightPos). The pass view
     *  eye is L - A and EVERY caster vertex (opaque VBO, cutout VBO, entity accum)
     *  enters as worldPos - A subtracted in DOUBLE before its float cast, so
     *  viewProj*(v-A) with eye L-A == R*(v-L): A cancels exactly, the stored depth
     *  contract is unchanged, the 1e5 float cancellation is removed. round((float)
     *  lightPos) == BlockShadowCache's block snap == round(getX), so this shared
     *  anchor stays in lockstep with the per-light cached block/cutout VBO. Exposed so
     *  the mods' entity arm (editor OccluderGeometryCapturer) subtracts the SAME A. */
    private static double currentOriginX, currentOriginY, currentOriginZ;
    public static double currentOriginX() { return currentOriginX; }
    public static double currentOriginY() { return currentOriginY; }
    public static double currentOriginZ() { return currentOriginZ; }

    // --- Reused matrix scratch (T1.3) ----------------------------------------
    // The spot projection depends only on (fov, far) and the point projection only
    // on far, but a bake runs many passes (every spot tile, every point face) —
    // recomputing perspective()'s trig per pass is wasted work when consecutive
    // lamps share parameters. These hold the last-built matrix + the params it was
    // built for; each begin*() copies the cached matrix into currentProj, rebuilding
    // only when params change (currentProj itself can't cache it — the other light
    // kind overwrites currentProj).
    private static final Matrix4f spotProj = new Matrix4f();
    private static float spotProjFov = Float.NaN;
    private static float spotProjFar = Float.NaN;
    private static final Matrix4f pointProj = new Matrix4f();
    private static float pointProjFar = Float.NaN;

    /** Reused raw-GL batch handed to {@link ShadowCasterSource#emitOccluder} (the
     *  render thread is single-threaded, so one instance is safe). It appends the
     *  source's world-space POSITION triangles into {@link #casterAccum}. */
    private static final RawOccluderBatch casterBatch = new RawOccluderBatch();

    private static long casterFailures;

    /** Monotonic failure stamp: a recovered partial draw must never be cached. */
    public static long casterFailures() { return casterFailures; }

    private ShadowRenderer()
    {}

    /** Call once at the start of a bake, before any begin*()/endPass(). Arms a
     *  fresh snapshot of the MC/Iris GL state on the first pass of this bake
     *  (see {@link #savePassState}), so the per-pass glGet* stalls collapse to
     *  one per bake. */
    public static void beginBake()
    {
        // A previous bake should have closed every pass. If a caller escaped
        // its normal failure handler, restore that pass before re-arming the
        // per-bake state snapshot.
        abortPass();

        passStateSaved = false;
        // Defensive: drop any unflushed pass triangles (endPass clears every pass).
        casterAccum.clear();
    }

    /**
     * Begin a spot depth pass into the live ({@code toStatic} false) or static
     * ({@code toStatic} true) atlas tile. {@code clear} false keeps the tile's
     * current depth — used for the dynamic-caster overlay drawn on top of a
     * static base just restored by {@link SpotlightDepthAtlas#copyStaticToLive}.
     */
    public static void beginSpot(int tile,
                                 double lpx, double lpy, double lpz,
                                 float ldx, float ldy, float ldz,
                                 float range, float outerDeg,
                                 boolean toStatic, boolean clear)
    {
        savePassState();
        try
        {

            // Light-relative bake anchor A = round((float) lightPos) (== BlockShadowCache's
            // block snap), and the pass view eye at L - A carried in float (the sub-block
            // residual). ldx/ldy/ldz stay float — direction is translation-invariant.
            currentOriginX = Math.round((float) lpx);
            currentOriginY = Math.round((float) lpy);
            currentOriginZ = Math.round((float) lpz);
            float ex = (float) (lpx - currentOriginX);
            float ey = (float) (lpy - currentOriginY);
            float ez = (float) (lpz - currentOriginZ);

            GL30.glBindFramebuffer(GL30.GL_DRAW_FRAMEBUFFER, SpotlightDepthAtlas.getFboId(toStatic));
            int px = SpotlightDepthAtlas.tilePixelX(tile);
            int py = SpotlightDepthAtlas.tilePixelY(tile);
            int ts = SpotlightDepthAtlas.tileSizePx(tile);
            GL11.glViewport(px, py, ts, ts);
            GL11.glEnable(GL11.GL_SCISSOR_TEST);
            GL11.glScissor(px, py, ts, ts);
            if (clear)
            {
                GL11.glDepthMask(true);
                GL11.glClearDepth(1.0);
                GL11.glClear(GL11.GL_DEPTH_BUFFER_BIT);
            }

            float fovDeg = Math.max(outerDeg, 1.0f);
            float far = Math.max(range, NEAR + 0.1f);
            if (fovDeg != spotProjFov || far != spotProjFar)
            {
                spotProj.identity().perspective((float) Math.toRadians(fovDeg), 1.0f, NEAR, far);
                spotProjFov = fovDeg;
                spotProjFar = far;
            }
            currentProj.set(spotProj);

            Vector3f up = pickStableUp(ldy);
            currentView.identity().lookAt(
                ex, ey, ez,
                ex + ldx, ey + ldy, ez + ldz,
                up.x, up.y, up.z
            );
        }
        catch (RuntimeException | Error t)
        {
            abortPass();
            throw t;
        }

    }

    /** Shrink the current spot tile's scissor to a partial sub-rect (the 1.1.3
     *  dyn-rect / partial-tile bake path). Call between {@link #beginSpot} (which
     *  set the full-tile scissor; the viewport/NDC mapping stays full-tile) and
     *  the caster draws; {@link #endPass} restores the caller's scissor. */
    public static void restrictScissorSpot(int tile, int localX, int localY, int w, int h)
    {
        int px = SpotlightDepthAtlas.tilePixelX(tile);
        int py = SpotlightDepthAtlas.tilePixelY(tile);
        GL11.glScissor(px + localX, py + localY, w, h);
    }

    /** DIAGNOSTIC (-Dirlite.dynRectDebug=true): clear the CURRENTLY SCISSORED
     *  depth region to the near plane so the partial-tile dyn rect shows up in
     *  the world as a fully-shadowed block. Call right after
     *  {@link #restrictScissorSpot}. */
    public static void debugFillScissoredDepth()
    {
        GL11.glClearDepth(0.0);
        GL11.glClear(GL11.GL_DEPTH_BUFFER_BIT);
        GL11.glClearDepth(1.0);
    }

    /**
     * Begin a point-cube face depth pass into the live or static atlas (see
     * {@link #beginSpot} for the {@code toStatic}/{@code clear} semantics; the
     * static base of a whole block is restored by
     * {@link PointDepthAtlas#copyStaticToLive}). {@code block} is the GLOBAL
     * point-atlas block index; each of the 6 cube faces renders into its own
     * pixel sub-rect of that block's atlas region.
     */
    public static void beginPointFace(int block, int face,
                                      double lpx, double lpy, double lpz,
                                      float radius,
                                      boolean toStatic, boolean clear)
    {
        savePassState();
        try
        {

            // Light-relative bake anchor A = round((float) lightPos) (== BlockShadowCache's
            // block snap), and the pass view eye at L - A carried in float (the sub-block
            // residual). The face dx/dy/dz below stay unit floats — translation-invariant.
            currentOriginX = Math.round((float) lpx);
            currentOriginY = Math.round((float) lpy);
            currentOriginZ = Math.round((float) lpz);
            float ex = (float) (lpx - currentOriginX);
            float ey = (float) (lpy - currentOriginY);
            float ez = (float) (lpz - currentOriginZ);

            GL30.glBindFramebuffer(GL30.GL_DRAW_FRAMEBUFFER, PointDepthAtlas.getFboId(toStatic));
            int px = PointDepthAtlas.facePixelX(block, face);
            int py = PointDepthAtlas.facePixelY(block, face);
            int fs = PointDepthAtlas.faceSizePx(block);
            GL11.glViewport(px, py, fs, fs);
            GL11.glEnable(GL11.GL_SCISSOR_TEST);
            GL11.glScissor(px, py, fs, fs);
            if (clear)
            {
                GL11.glDepthMask(true);
                GL11.glClearDepth(1.0);
                GL11.glClear(GL11.GL_DEPTH_BUFFER_BIT);
            }

            float far = Math.max(radius, NEAR + 0.1f);
            if (far != pointProjFar)
            {
                pointProj.identity().perspective((float) Math.toRadians(90.0), 1.0f, NEAR, far);
                pointProjFar = far;
            }
            currentProj.set(pointProj);

            float dx, dy, dz, ux, uy, uz;
            switch (face)
            {
                case 0:  dx =  1; dy =  0; dz =  0; ux = 0; uy = -1; uz =  0; break; // +X
                case 1:  dx = -1; dy =  0; dz =  0; ux = 0; uy = -1; uz =  0; break; // -X
                case 2:  dx =  0; dy =  1; dz =  0; ux = 0; uy =  0; uz =  1; break; // +Y
                case 3:  dx =  0; dy = -1; dz =  0; ux = 0; uy =  0; uz = -1; break; // -Y
                case 4:  dx =  0; dy =  0; dz =  1; ux = 0; uy = -1; uz =  0; break; // +Z
                default: dx =  0; dy =  0; dz = -1; ux = 0; uy = -1; uz =  0; break; // -Z
            }

            currentView.identity().lookAt(
                ex, ey, ez,
                ex + dx, ey + dy, ez + dz,
                ux, uy, uz
            );
        }
        catch (RuntimeException | Error t)
        {
            abortPass();
            throw t;
        }

    }

    // --- Batched caster pass (T2.2) ------------------------------------------
    // A pass brackets its casters with
    //   beginCasterBatch() -> emitCaster()* -> endCasterBatch()
    // and flushes ONCE (in endPass): each caster APPENDS its world-space POSITION
    // triangles into casterAccum, and a single glDrawArrays at pass end submits
    // them all, so a pass costs at most one caster draw regardless of subject count.

    /** Triangles (POSITION xyz) accumulated for the current pass; uploaded + drawn
     *  once in {@link #endPass} so many casters cost a single draw + state snapshot. */
    private static final FloatArrayList casterAccum = new FloatArrayList();
    private static int casterScratchVbo = 0;
    /** Persistent off-heap upload buffer, grown on demand and reused across passes
     *  (a moving caster flushes every dynamic pass / cube face — a fresh
     *  memAllocFloat each time would be per-frame native churn). Freed by
     *  {@link #releaseScratch}. */
    private static FloatBuffer casterScratch;
    private static int casterScratchCap;

    /** Open a batched caster pass. Enters baking mode so a mod's light-form
     *  renderers skip re-registration while a caster's model is captured during
     *  the bake. Casters are emitted with {@link #emitCaster} and flushed by
     *  {@link #endPass}. No-op outside a begin*()/endPass() bracket. */
    public static void beginCasterBatch()
    {
        if (!inPass)
        {
            return;
        }
        ShadowBakeState.setBaking(true);
    }

    /** Shared per-caster wrapper around {@link ShadowCasterSource#emitOccluder}.
     *  Owns the per-caster run isolation (INVARIANT 4): the source appends one
     *  caster's world-space POSITION triangles into the shared accumulator (via
     *  {@link RawOccluderBatch}); if it throws mid-append the run is rewound to the
     *  start-of-this-caster mark here, so the broken caster's partial vertices can
     *  never fuse with the next caster's into a garbage triangle. The source NEVER
     *  flushes or catches its own throw.
     *
     *  <p>The raw-GL caster path is INVARIANT-1 exempt: it appends world-space
     *  geometry drawn through the depth program's explicit uViewProj (never MC's
     *  live modelview, which the hot path no longer touches), so no per-caster
     *  matrix / depth-state re-establish is needed (unlike the 1.21.4 Immediate
     *  backend). See {@code docs/shadow-caster-seam-spec.md} INV-1 EXEMPTION. */
    public static void emitCaster(ShadowCasterSource source, Object caster, int casterType, float tickDelta)
    {
        if (source == null || caster == null || !inPass || glBroken)
        {
            return;
        }
        if (!glInit)
        {
            initGl();
        }
        if (glBroken || program == 0)
        {
            return;
        }

        casterBatch.bind(casterAccum);
        casterBatch.mark();
        try
        {
            source.emitOccluder(caster, casterType, tickDelta, casterBatch);
        }
        catch (Throwable t)
        {
            casterFailures++;
            // The caster threw mid-append: rewind its partial run to the mark so it
            // ends at a whole-caster boundary instead of merging into the next one.
            casterBatch.terminateRun(currentView, currentProj);
        }
    }

    /** Close a batched caster pass: leave baking mode. The accumulated casters are
     *  drawn (once) by {@link #endPass}, which still holds this pass's
     *  view/proj + bound FBO. */
    public static void endCasterBatch()
    {
        ShadowBakeState.setBaking(false);
    }

    /** Draw the caster silhouette triangles accumulated this pass (if any) with the
     *  position depth program, then clear the accumulator. Called by {@link #endPass}
     *  before the GL state is restored (they share the pass's currentView/currentProj,
     *  still set). Mirrors {@link #drawOpaqueBlocks}'s state discipline: snapshot
     *  every global bit the draw mutates and restore it exactly. */
    private static void flushCasters()
    {
        if (casterAccum.isEmpty() || glBroken || program == 0)
        {
            return;
        }
        int floats = casterAccum.size();
        int verts = floats / 3;

        int prevProgram = GL11.glGetInteger(GL20.GL_CURRENT_PROGRAM);
        int prevVao = GL11.glGetInteger(GL30.GL_VERTEX_ARRAY_BINDING);
        int prevArrayBuf = GL11.glGetInteger(GL15.GL_ARRAY_BUFFER_BINDING);
        boolean prevDepthTest = GL11.glIsEnabled(GL11.GL_DEPTH_TEST);
        int prevDepthFunc = GL11.glGetInteger(GL11.GL_DEPTH_FUNC);
        boolean prevDepthMask = GL11.glGetBoolean(GL11.GL_DEPTH_WRITEMASK);
        boolean prevCull = GL11.glIsEnabled(GL11.GL_CULL_FACE);

        // Grow-only persistent upload buffer (no per-pass alloc/free).
        if (casterScratch == null || floats > casterScratchCap)
        {
            if (casterScratch != null)
            {
                MemoryUtil.memFree(casterScratch);
            }
            casterScratchCap = Math.max(floats, 4096);
            casterScratch = MemoryUtil.memAllocFloat(casterScratchCap);
        }
        FloatBuffer fb = casterScratch;
        try
        {
            fb.clear();
            fb.put(casterAccum.elements(), 0, floats);
            fb.flip();

            GL11.glEnable(GL11.GL_DEPTH_TEST);
            GL11.glDepthFunc(GL11.GL_LEQUAL);
            GL11.glDepthMask(true);
            // Cull OFF: the model captures both windings; the depth test keeps the
            // nearest (light-facing) surface -> a tight, correct silhouette.
            GL11.glDisable(GL11.GL_CULL_FACE);

            GL20.glUseProgram(program);
            currentProj.mul(currentView, viewProj);
            try (org.lwjgl.system.MemoryStack stack = org.lwjgl.system.MemoryStack.stackPush())
            {
                FloatBuffer mb = stack.mallocFloat(16);
                viewProj.get(mb);
                GL20.glUniformMatrix4fv(uViewProjLoc, false, mb);
            }

            if (casterScratchVbo == 0)
            {
                casterScratchVbo = GL15.glGenBuffers();
            }
            GL30.glBindVertexArray(vao);
            GL15.glBindBuffer(GL15.GL_ARRAY_BUFFER, casterScratchVbo);
            GL15.glBufferData(GL15.GL_ARRAY_BUFFER, fb, GL15.GL_STREAM_DRAW);
            GL20.glEnableVertexAttribArray(0);
            GL20.glVertexAttribPointer(0, 3, GL11.GL_FLOAT, false, 12, 0L);

            GL11.glDrawArrays(GL11.GL_TRIANGLES, 0, verts);
        }
        catch (Throwable t)
        {
            if (!blockErrorLogged)
            {
                blockErrorLogged = true;
                System.err.println("[irl-core] caster shadow bake failed: " + t);
                t.printStackTrace();
            }
        }
        finally
        {
            try
            {
                restoreGlStateStep(() -> GL15.glBindBuffer(GL15.GL_ARRAY_BUFFER, prevArrayBuf));
                restoreGlStateStep(() -> GL30.glBindVertexArray(prevVao));
                restoreGlStateStep(() -> GL20.glUseProgram(prevProgram));
                restoreGlStateStep(() -> GL11.glDepthMask(prevDepthMask));
                restoreGlStateStep(() -> GL11.glDepthFunc(prevDepthFunc));
                restoreGlStateStep(() -> { if (prevDepthTest) { GL11.glEnable(GL11.GL_DEPTH_TEST); } else { GL11.glDisable(GL11.GL_DEPTH_TEST); } });
                restoreGlStateStep(() -> { if (prevCull) { GL11.glEnable(GL11.GL_CULL_FACE); } else { GL11.glDisable(GL11.GL_CULL_FACE); } });
            }
            finally
            {
                casterAccum.clear();
            }
        }
    }

    /** Free the persistent caster scratch buffer + its VBO (called on shaders-off,
     *  alongside the depth textures). Both lazily re-create on the next bake. */
    public static void releaseScratch()
    {
        if (casterScratch != null)
        {
            MemoryUtil.memFree(casterScratch);
            casterScratch = null;
            casterScratchCap = 0;
        }
        if (casterScratchVbo != 0)
        {
            GL15.glDeleteBuffers(casterScratchVbo);
            casterScratchVbo = 0;
        }
    }

    // --- Block-shadow depth bake (raw-GL) -------------------------------------

    /** Shared POSITION depth program (opaque blocks + casters) + its uniform/VAO,
     *  lazily created on the first bake. */
    private static int program = 0;
    private static int uViewProjLoc = -1;
    private static int vao = 0;
    private static boolean glInit = false;
    private static boolean glBroken = false;

    /** Per-light cached VBO of triangle vertices (POSITION). Rebuilt only when the
     *  block list instance changes ({@link BlockShadowCache} returns a stable
     *  instance until a block in range changes), so static lamps just redraw. */
    private static final Long2ObjectOpenHashMap<List<BlockShadowEntry>> vboList = new Long2ObjectOpenHashMap<>();
    private static final Long2IntOpenHashMap vboId = new Long2IntOpenHashMap();
    private static final Long2IntOpenHashMap vboVertCount = new Long2IntOpenHashMap();

    // --- Cutout (textured, alpha-tested) block depth program + per-light cache ---
    // Cutout blocks (leaves / glass panes / iron bars / doors / foliage; shape ==
    // null in the entry) are baked from their BakedModel quads through a second
    // depth program that samples the block atlas and discards transparent texels,
    // so light passes through the holes. Vertices are interleaved POSITION(3) +
    // UV(2) (stride 20). Per-light VBO cached + rebuilt on a list-instance change,
    // exactly like the opaque path; evicted alongside it in retainBlockVbos. The
    // CUTOUT and CUTOUT_MIPPED layers of the 1.21.4 line collapse into the single
    // BlockRenderLayer.CUTOUT the collector classifies, so there is ONE cutout VBO
    // per light here (the 1.21.4 line kept two, one per layer object).
    private static int cutoutProgram = 0;
    private static int cutoutViewProjLoc = -1;
    private static int cutoutAtlasLoc = -1;
    private static int cutoutVao = 0;
    private static final Long2ObjectOpenHashMap<List<BlockShadowEntry>> cutoutList = new Long2ObjectOpenHashMap<>();
    private static final Long2IntOpenHashMap cutoutVboId = new Long2IntOpenHashMap();
    private static final Long2IntOpenHashMap cutoutVertCount = new Long2IntOpenHashMap();
    private static final RandomSource cutoutRandom = RandomSource.create();
    private static boolean cutoutErrorLogged = false;

    private static boolean blockErrorLogged = false;
    private static boolean stateRestoreErrorLogged = false;

    /** Attempt every state restore independently: one failed GL call must not
     *  prevent the remaining caller state from being put back. */
    private static void restoreGlStateStep(Runnable restore)
    {
        try
        {
            restore.run();
        }
        catch (Throwable t)
        {
            if (!stateRestoreErrorLogged)
            {
                stateRestoreErrorLogged = true;
                System.err.println("[irl-core] shadow GL state restore failed: " + t);
                t.printStackTrace();
            }
        }
    }

    /**
     * Render a light's block occluders into the currently-bound depth FBO, between
     * a begin*()/endPass() bracket and after the entity casters. Opaque-shape
     * blocks are drawn as their {@link BlockShadowEntry#shape} AABB triangles;
     * cutout blocks (shape == null) are drawn from their real MC-tessellated
     * textured geometry with alpha discard (see {@link #renderBlocksDepthCutout}).
     * Both use the per-light view/proj.
     */
    public static void renderBlocksDepth(long id, List<BlockShadowEntry> blocks)
    {
        if (blocks == null || blocks.isEmpty() || !inPass || glBroken)
        {
            return;
        }
        if (!glInit)
        {
            initGl();
        }
        if (glBroken)
        {
            return;
        }
        drawOpaqueBlocks(id, blocks);
        renderBlocksDepthCutout(id, blocks);
    }

    /** Draw the opaque-shape (non-cutout) blocks of a light as triangle AABBs. */
    private static void drawOpaqueBlocks(long id, List<BlockShadowEntry> blocks)
    {
        if (program == 0)
        {
            return;
        }

        // (Re)build this light's VBO only when its block list instance changed.
        // Keyed on the list instance alone (not vbo != 0): buildVbo stores the
        // instance even when there is no shaped geometry (a cutout-only lamp), so
        // gating on vbo would re-tessellate that empty lamp every frame.
        int verts;
        if (vboList.get(id) != blocks)
        {
            verts = buildVbo(id, blocks);
        }
        else
        {
            verts = vboVertCount.get(id);
        }
        if (verts <= 0)
        {
            return;
        }
        int vbo = vboId.get(id);

        // Snapshot every global GL state the draw mutates, then restore it — MC's
        // GlStateManager doesn't cache program/VAO/buffer, and the FBO/viewport are
        // restored by endPass, so this keeps the surrounding renderer consistent.
        int prevProgram = GL11.glGetInteger(GL20.GL_CURRENT_PROGRAM);
        int prevVao = GL11.glGetInteger(GL30.GL_VERTEX_ARRAY_BINDING);
        int prevArrayBuf = GL11.glGetInteger(GL15.GL_ARRAY_BUFFER_BINDING);
        boolean prevDepthTest = GL11.glIsEnabled(GL11.GL_DEPTH_TEST);
        int prevDepthFunc = GL11.glGetInteger(GL11.GL_DEPTH_FUNC);
        boolean prevDepthMask = GL11.glGetBoolean(GL11.GL_DEPTH_WRITEMASK);
        boolean prevCull = GL11.glIsEnabled(GL11.GL_CULL_FACE);

        try
        {
            GL11.glEnable(GL11.GL_DEPTH_TEST);
            GL11.glDepthFunc(GL11.GL_LEQUAL);
            GL11.glDepthMask(true);
            // Cull OFF: both faces of every box triangle write depth and the depth
            // test keeps the nearest (light-facing) surface — tight, correct block
            // silhouettes regardless of winding.
            GL11.glDisable(GL11.GL_CULL_FACE);

            GL20.glUseProgram(program);
            currentProj.mul(currentView, viewProj);
            try (org.lwjgl.system.MemoryStack stack = org.lwjgl.system.MemoryStack.stackPush())
            {
                FloatBuffer fb = stack.mallocFloat(16);
                viewProj.get(fb);
                GL20.glUniformMatrix4fv(uViewProjLoc, false, fb);
            }

            GL30.glBindVertexArray(vao);
            GL15.glBindBuffer(GL15.GL_ARRAY_BUFFER, vbo);
            GL20.glEnableVertexAttribArray(0);
            GL20.glVertexAttribPointer(0, 3, GL11.GL_FLOAT, false, 12, 0L);

            GL11.glDrawArrays(GL11.GL_TRIANGLES, 0, verts);
        }
        catch (Throwable t)
        {
            if (!blockErrorLogged)
            {
                blockErrorLogged = true;
                System.err.println("[irl-core] block shadow bake failed: " + t);
                t.printStackTrace();
            }
        }
        finally
        {
            // Restore each captured bit even if an earlier GL call fails.
            restoreGlStateStep(() -> GL15.glBindBuffer(GL15.GL_ARRAY_BUFFER, prevArrayBuf));
            restoreGlStateStep(() -> GL30.glBindVertexArray(prevVao));
            restoreGlStateStep(() -> GL20.glUseProgram(prevProgram));
            restoreGlStateStep(() -> GL11.glDepthMask(prevDepthMask));
            restoreGlStateStep(() -> GL11.glDepthFunc(prevDepthFunc));
            restoreGlStateStep(() -> { if (prevDepthTest) { GL11.glEnable(GL11.GL_DEPTH_TEST); } else { GL11.glDisable(GL11.GL_DEPTH_TEST); } });
            restoreGlStateStep(() -> { if (prevCull) { GL11.glEnable(GL11.GL_CULL_FACE); } else { GL11.glDisable(GL11.GL_CULL_FACE); } });
        }
    }

    /** Build (or rebuild) the per-light triangle VBO from the block shape AABBs.
     *  Returns the vertex count (0 if there is no shaped geometry). */
    private static int buildVbo(long id, List<BlockShadowEntry> blocks)
    {
        // Count boxes first (cheap AABB iteration) to size one allocation.
        int[] boxes = {0};
        for (int i = 0, n = blocks.size(); i < n; i++)
        {
            BlockShadowEntry e = blocks.get(i);
            if (e != null && e.shape != null)
            {
                e.shape.forAllBoxes((a, b, c, d, f, g) -> boxes[0]++);
            }
        }
        int boxCount = boxes[0];
        if (boxCount == 0)
        {
            // No shaped geometry (e.g. only cutout entries): drop any VBO.
            releaseBlockVbo(id);
            vboList.put(id, blocks);
            vboVertCount.put(id, 0);
            return 0;
        }

        int vertCount = boxCount * 36; // 6 faces * 2 tris * 3 verts
        FloatBuffer fb = MemoryUtil.memAllocFloat(vertCount * 3);
        try
        {
            for (int i = 0, n = blocks.size(); i < n; i++)
            {
                BlockShadowEntry e = blocks.get(i);
                if (e == null || e.shape == null)
                {
                    continue;
                }
                // worldPos - A subtracted in DOUBLE before the float cast: (int pos) -
                // (integral double A) + (double shape) -> float is exact, and the cache
                // bakes A in — the block-list instance changes exactly when the snap/A
                // changes (line 493 gate), so a cached VBO's A always equals the pass
                // eye's A. Mirrors main QuadBoxConsumer x1 = (float)(ox - originX + minX).
                final int ox = e.pos.getX(), oy = e.pos.getY(), oz = e.pos.getZ();
                final double ax = currentOriginX, ay = currentOriginY, az = currentOriginZ;
                e.shape.forAllBoxes((minX, minY, minZ, maxX, maxY, maxZ) ->
                    emitBox(fb,
                        (float) (ox - ax + minX), (float) (oy - ay + minY), (float) (oz - az + minZ),
                        (float) (ox - ax + maxX), (float) (oy - ay + maxY), (float) (oz - az + maxZ)));
            }
            fb.flip();

            int vbo = vboId.get(id);
            boolean createdVbo = vbo == 0;
            if (createdVbo)
            {
                vbo = GL15.glGenBuffers();
            }
            boolean uploaded = false;
            try
            {
                int prevArrayBuf = GL11.glGetInteger(GL15.GL_ARRAY_BUFFER_BINDING);
                try
                {
                    GL15.glBindBuffer(GL15.GL_ARRAY_BUFFER, vbo);
                    GL15.glBufferData(GL15.GL_ARRAY_BUFFER, fb, GL15.GL_STATIC_DRAW);
                    uploaded = true;
                }
                finally
                {
                    GL15.glBindBuffer(GL15.GL_ARRAY_BUFFER, prevArrayBuf);
                }
            }
            finally
            {
                // Do not publish a newly-created name until its storage upload
                // succeeds; otherwise a failed first upload leaves an orphaned
                // buffer that the cache sweep cannot see.
                if (createdVbo)
                {
                    if (uploaded)
                    {
                        vboId.put(id, vbo);
                    }
                    else if (vbo != 0)
                    {
                        GL15.glDeleteBuffers(vbo);
                    }
                }
            }

            vboList.put(id, blocks);
            vboVertCount.put(id, vertCount);
            return vertCount;
        }
        finally
        {
            MemoryUtil.memFree(fb);
        }
    }

    /** Append one axis-aligned box as 12 triangles (36 verts, POSITION only).
     *  Winding is irrelevant — culling is disabled during the bake. */
    private static void emitBox(FloatBuffer b,
                                float x1, float y1, float z1,
                                float x2, float y2, float z2)
    {
        // -X
        tri(b, x1,y1,z1, x1,y1,z2, x1,y2,z2);  tri(b, x1,y1,z1, x1,y2,z2, x1,y2,z1);
        // +X
        tri(b, x2,y1,z1, x2,y2,z1, x2,y2,z2);  tri(b, x2,y1,z1, x2,y2,z2, x2,y1,z2);
        // -Y
        tri(b, x1,y1,z1, x2,y1,z1, x2,y1,z2);  tri(b, x1,y1,z1, x2,y1,z2, x1,y1,z2);
        // +Y
        tri(b, x1,y2,z1, x1,y2,z2, x2,y2,z2);  tri(b, x1,y2,z1, x2,y2,z2, x2,y2,z1);
        // -Z
        tri(b, x1,y1,z1, x1,y2,z1, x2,y2,z1);  tri(b, x1,y1,z1, x2,y2,z1, x2,y1,z1);
        // +Z
        tri(b, x1,y1,z2, x2,y1,z2, x2,y2,z2);  tri(b, x1,y1,z2, x2,y2,z2, x1,y2,z2);
    }

    private static void tri(FloatBuffer b,
                            float ax, float ay, float az,
                            float bx, float by, float bz,
                            float cx, float cy, float cz)
    {
        b.put(ax).put(ay).put(az);
        b.put(bx).put(by).put(bz);
        b.put(cx).put(cy).put(cz);
    }

    // --- Cutout block depth bake (textured, alpha-tested) ---------------------

    /**
     * Draw a light's cutout blocks from baked model-part quads
     * (see {@link #captureCutoutBlockTris}), sampling the block atlas and discarding
     * transparent texels so light passes through (lacy leaf / grate / glass-pane
     * shadows). Per-light VBO cached like the opaque path. Called from
     * {@link #renderBlocksDepth} within a pass.
     */
    private static void renderBlocksDepthCutout(long id, List<BlockShadowEntry> blocks)
    {
        if (cutoutProgram == 0)
        {
            return;
        }

        // Keyed on the list instance alone (see drawOpaqueBlocks): buildCutoutVbo
        // stores the instance even with zero quads, so this won't re-tessellate an
        // empty lamp every frame.
        int verts;
        if (cutoutList.get(id) != blocks)
        {
            verts = buildCutoutVbo(id, blocks);
        }
        else
        {
            verts = cutoutVertCount.get(id);
        }
        if (verts <= 0)
        {
            return;
        }
        int vbo = cutoutVboId.get(id);

        int atlas = atlasGlId();
        int atlasSampler = atlasSamplerGlId();
        if (atlas == 0 || atlasSampler == 0)
        {
            return; // atlas texture not ready this frame
        }

        // Snapshot every global GL bit the draw mutates (program/VAO/buffer, depth,
        // cull, blend, active texture unit + its 2D binding) and restore it, like
        // the opaque path — keeps the surrounding renderer's GL state consistent.
        int prevProgram = GL11.glGetInteger(GL20.GL_CURRENT_PROGRAM);
        int prevVao = GL11.glGetInteger(GL30.GL_VERTEX_ARRAY_BINDING);
        int prevArrayBuf = GL11.glGetInteger(GL15.GL_ARRAY_BUFFER_BINDING);
        boolean prevDepthTest = GL11.glIsEnabled(GL11.GL_DEPTH_TEST);
        int prevDepthFunc = GL11.glGetInteger(GL11.GL_DEPTH_FUNC);
        boolean prevDepthMask = GL11.glGetBoolean(GL11.GL_DEPTH_WRITEMASK);
        boolean prevCull = GL11.glIsEnabled(GL11.GL_CULL_FACE);
        boolean prevBlend = GL11.glIsEnabled(GL11.GL_BLEND);
        int prevActiveTex = GL11.glGetInteger(GL13.GL_ACTIVE_TEXTURE);
        GL13.glActiveTexture(GL13.GL_TEXTURE0);
        int prevTex0 = GL11.glGetInteger(GL11.GL_TEXTURE_BINDING_2D);
        int prevSampler0 = GL11.glGetInteger(GL33.GL_SAMPLER_BINDING);

        try
        {
            GL11.glEnable(GL11.GL_DEPTH_TEST);
            GL11.glDepthFunc(GL11.GL_LEQUAL);
            GL11.glDepthMask(true);
            GL11.glDisable(GL11.GL_CULL_FACE);
            GL11.glDisable(GL11.GL_BLEND);

            GL20.glUseProgram(cutoutProgram);
            currentProj.mul(currentView, viewProj);
            try (org.lwjgl.system.MemoryStack stack = org.lwjgl.system.MemoryStack.stackPush())
            {
                FloatBuffer fb = stack.mallocFloat(16);
                viewProj.get(fb);
                GL20.glUniformMatrix4fv(cutoutViewProjLoc, false, fb);
            }
            GL11.glBindTexture(GL11.GL_TEXTURE_2D, atlas);
            GL33.glBindSampler(0, atlasSampler);
            GL20.glUniform1i(cutoutAtlasLoc, 0);

            GL30.glBindVertexArray(cutoutVao);
            GL15.glBindBuffer(GL15.GL_ARRAY_BUFFER, vbo);
            GL20.glEnableVertexAttribArray(0);
            GL20.glEnableVertexAttribArray(1);
            GL20.glVertexAttribPointer(0, 3, GL11.GL_FLOAT, false, 20, 0L);
            GL20.glVertexAttribPointer(1, 2, GL11.GL_FLOAT, false, 20, 12L);

            GL11.glDrawArrays(GL11.GL_TRIANGLES, 0, verts);
        }
        catch (Throwable t)
        {
            if (!cutoutErrorLogged)
            {
                cutoutErrorLogged = true;
                System.err.println("[irl-core] cutout block shadow bake failed: " + t);
                t.printStackTrace();
            }
        }
        finally
        {
            restoreGlStateStep(() -> GL11.glBindTexture(GL11.GL_TEXTURE_2D, prevTex0));
            restoreGlStateStep(() -> GL33.glBindSampler(0, prevSampler0));
            restoreGlStateStep(() -> GL13.glActiveTexture(prevActiveTex));
            restoreGlStateStep(() -> GL15.glBindBuffer(GL15.GL_ARRAY_BUFFER, prevArrayBuf));
            restoreGlStateStep(() -> GL30.glBindVertexArray(prevVao));
            restoreGlStateStep(() -> GL20.glUseProgram(prevProgram));
            restoreGlStateStep(() -> GL11.glDepthMask(prevDepthMask));
            restoreGlStateStep(() -> GL11.glDepthFunc(prevDepthFunc));
            restoreGlStateStep(() -> { if (prevDepthTest) { GL11.glEnable(GL11.GL_DEPTH_TEST); } else { GL11.glDisable(GL11.GL_DEPTH_TEST); } });
            restoreGlStateStep(() -> { if (prevCull) { GL11.glEnable(GL11.GL_CULL_FACE); } else { GL11.glDisable(GL11.GL_CULL_FACE); } });
            restoreGlStateStep(() -> { if (prevBlend) { GL11.glEnable(GL11.GL_BLEND); } else { GL11.glDisable(GL11.GL_BLEND); } });
        }
    }

    /** Build (or rebuild) a light's cutout VBO (interleaved POSITION(3)+UV(2),
     *  stride 20) from cutout model-part quads and their baked atlas UVs,
     *  translated into light-relative world coordinates by
     *  {@link #captureCutoutBlockTris}. Returns the vertex count (0 if there is
     *  no cutout geometry). */
    private static int buildCutoutVbo(long id, List<BlockShadowEntry> blocks)
    {
        Minecraft mc = Minecraft.getInstance();
        ClientLevel world = mc != null ? mc.level : null;
        BlockStateModelSet models = mc != null ? mc.getModelManager().getBlockStateModelSet() : null;
        if (world == null || models == null) { releaseCutoutVbo(id); cutoutList.put(id, blocks); cutoutVertCount.put(id, 0); return 0; }

        FloatArrayList data = new FloatArrayList();
        // Suppress the mod's per-frame light re-registration while a ModelBlock's
        // real render runs inside the capture (same gate the opaque path relies on
        // implicitly via the caster batch). Cleared in the finally.
        boolean wasBaking = ShadowBakeState.isBaking();
        ShadowBakeState.setBaking(true);
        try
        {
            for (int i = 0, n = blocks.size(); i < n; i++)
            {
                BlockShadowEntry e = blocks.get(i);
                if (e == null || !e.cutout)
                {
                    continue;
                }
                BlockPos p = e.pos;
                try
                {
                    BlockState state = world.getBlockState(p);
                    if (state.isAir())
                    {
                        continue;
                    }
                    // Real MC tessellation (correct atlas UVs) -> world-space
                    // POSITION+UV triangles in the stride-20 layout this buffer
                    // expects.
                    float[] tris = captureCutoutBlockTris(models, world, p, state, cutoutRandom);
                    if (tris.length > 0)
                    {
                        data.addElements(data.size(), tris);
                    }
                }
                catch (Throwable t)
                {
                    // skip a single broken block, keep tessellating the rest
                }
            }
        }
        finally
        {
            ShadowBakeState.setBaking(wasBaking);
        }

        int floats = data.size();
        if (floats == 0)
        {
            releaseCutoutVbo(id);
            cutoutList.put(id, blocks);
            cutoutVertCount.put(id, 0);
            return 0;
        }
        int verts = floats / 5;

        FloatBuffer fb = MemoryUtil.memAllocFloat(floats);
        try
        {
            fb.put(data.elements(), 0, floats);
            fb.flip();

            int vbo = cutoutVboId.get(id);
            boolean createdVbo = vbo == 0;
            if (createdVbo)
            {
                vbo = GL15.glGenBuffers();
            }
            boolean uploaded = false;
            try
            {
                int prevArrayBuf = GL11.glGetInteger(GL15.GL_ARRAY_BUFFER_BINDING);
                try
                {
                    GL15.glBindBuffer(GL15.GL_ARRAY_BUFFER, vbo);
                    GL15.glBufferData(GL15.GL_ARRAY_BUFFER, fb, GL15.GL_STATIC_DRAW);
                    uploaded = true;
                }
                finally
                {
                    GL15.glBindBuffer(GL15.GL_ARRAY_BUFFER, prevArrayBuf);
                }
            }
            finally
            {
                // Keep failed first uploads out of the cache and release their
                // temporary GL name so cache eviction cannot miss the buffer.
                if (createdVbo)
                {
                    if (uploaded)
                    {
                        cutoutVboId.put(id, vbo);
                    }
                    else if (vbo != 0)
                    {
                        GL15.glDeleteBuffers(vbo);
                    }
                }
            }

            cutoutList.put(id, blocks);
            cutoutVertCount.put(id, verts);
            return verts;
        }
        finally
        {
            MemoryUtil.memFree(fb);
        }
    }

    /**
     * Tessellate one cutout block as world-space POSITION+UV triangles (5 floats
     * per vertex) through MC's real {@link BlockRenderDispatcher#renderBatched} into a
     * recording {@link Capture} consumer, then triangulate the captured quads.
     * The 1.21.4 line did this straight into a {@code BufferBuilder}; on 1.21.11
     * that immediate path is gone, so the tessellation is captured to a flat float
     * array and rasterized by {@link #renderBlocksDepthCutout}'s atlas-sampling,
     * alpha-discarding program. Empty array on any failure (no shadow for that
     * block, never a crashed bake).
     */
    private static float[] captureCutoutBlockTris(BlockStateModelSet models, ClientLevel world, BlockPos pos, BlockState state, RandomSource random)
    {
        if (models == null || world == null || state == null) return EMPTY_TRIS;
        try {
            BlockStateModel model = models.get(state);
            if (model == null) return EMPTY_TRIS;
            random.setSeed(state.getSeed(pos));
            List<BlockStateModelPart> parts = new ArrayList<>();
            model.collectParts(random, parts);
            FloatArrayList out = new FloatArrayList();
            float ox = (float)(pos.getX() - currentOriginX), oy = (float)(pos.getY() - currentOriginY), oz = (float)(pos.getZ() - currentOriginZ);
            for (BlockStateModelPart part : parts) {
                // null = unculled quads; directional buckets contain the faces
                // culled against neighbouring blocks. Capture both or block
                // shadows lose the majority of their visible side surfaces.
                for (net.minecraft.core.Direction direction : new net.minecraft.core.Direction[]{
                        null,
                        net.minecraft.core.Direction.DOWN,
                        net.minecraft.core.Direction.UP,
                        net.minecraft.core.Direction.NORTH,
                        net.minecraft.core.Direction.SOUTH,
                        net.minecraft.core.Direction.WEST,
                        net.minecraft.core.Direction.EAST}) {
                    for (BakedQuad quad : part.getQuads(direction)) {
                        if (quad.materialInfo().layer() != ChunkSectionLayer.CUTOUT) continue;
                        for (int vertex : new int[]{0,1,2,0,2,3}) {
                            org.joml.Vector3fc p = quad.position(vertex);
                            long uv = quad.packedUV(vertex);
                            out.add(ox+p.x()); out.add(oy+p.y()); out.add(oz+p.z());
                            out.add(Float.intBitsToFloat((int)uv));
                            out.add(Float.intBitsToFloat((int)(uv >>> 32)));
                        }
                    }
                }
            }
            return out.toFloatArray();
        } catch (Throwable t) { return EMPTY_TRIS; }
    }

    private static final float[] EMPTY_TRIS = new float[0];
    /** Raw OpenGL sampler id for the block atlas, or 0 if unavailable. */
    private static int atlasSamplerGlId()
    {
        Minecraft mc = Minecraft.getInstance();
        if (mc == null) return 0;
        try
        {
            net.minecraft.client.renderer.texture.AbstractTexture atlas =
                mc.getTextureManager().getTexture(TextureAtlas.LOCATION_BLOCKS);
            if (atlas == null || atlas.getSampler() == null || atlas.getSampler().isClosed()) return 0;
            return atlas.getSampler() instanceof GlSampler glSampler ? glSampler.getId() : 0;
        }
        catch (Throwable t)
        {
            return 0;
        }
    }

    /** Raw GL id of the block atlas texture, or 0 if it is not allocated yet. */
    private static int atlasGlId()
    {
        Minecraft mc = Minecraft.getInstance();
        if (mc == null)
        {
            return 0;
        }

        // The shadow backend below uses raw OpenGL calls, so only an OpenGL
        // RenderPearl texture can be sampled here. Vulkan textures deliberately
        // return 0 rather than being treated as GL object names.
        try
        {
            net.minecraft.client.renderer.texture.AbstractTexture atlas =
                mc.getTextureManager().getTexture(TextureAtlas.LOCATION_BLOCKS);
            if (atlas == null || atlas.getTexture() == null || atlas.getTexture().isClosed())
            {
                return 0;
            }
            return atlas.getTexture() instanceof GlTexture glTexture ? glTexture.glId() : 0;
        }
        catch (Throwable t)
        {
            return 0;
        }
    }

    private static void releaseCutoutVbo(long id)
    {
        int vbo = cutoutVboId.remove(id);
        if (vbo != 0)
        {
            GL15.glDeleteBuffers(vbo);
        }
        cutoutList.remove(id);
        cutoutVertCount.remove(id);
    }

    /** Lazily compile the depth programs + create the shared VAOs. */
    private static void initGl()
    {
        glInit = true;
        int vs = 0;
        int fs = 0;
        int prog = 0;
        int cvs = 0;
        int cfs = 0;
        int cprog = 0;
        int newVao = 0;
        int newCutoutVao = 0;
        try
        {
            vs = compile(GL20.GL_VERTEX_SHADER,
                "#version 150\n" +
                "in vec3 aPos;\n" +
                "uniform mat4 uViewProj;\n" +
                "void main() { gl_Position = uViewProj * vec4(aPos, 1.0); }\n");
            fs = compile(GL20.GL_FRAGMENT_SHADER,
                "#version 150\n" +
                "void main() {}\n");

            prog = GL20.glCreateProgram();
            GL20.glAttachShader(prog, vs);
            GL20.glAttachShader(prog, fs);
            GL20.glBindAttribLocation(prog, 0, "aPos");
            GL20.glLinkProgram(prog);
            if (GL20.glGetProgrami(prog, GL20.GL_LINK_STATUS) == GL11.GL_FALSE)
            {
                throw new IllegalStateException("link: " + GL20.glGetProgramInfoLog(prog));
            }
            GL20.glDeleteShader(vs);
            vs = 0;
            GL20.glDeleteShader(fs);
            fs = 0;

            newVao = GL30.glGenVertexArrays();

            // Cutout depth program: positions + atlas UVs, discards transparent
            // texels so light passes through the holes in the texture.
            cvs = compile(GL20.GL_VERTEX_SHADER,
                "#version 150\n" +
                "in vec3 aPos;\n" +
                "in vec2 aUV;\n" +
                "uniform mat4 uViewProj;\n" +
                "out vec2 vUV;\n" +
                "void main() { vUV = aUV; gl_Position = uViewProj * vec4(aPos, 1.0); }\n");
            cfs = compile(GL20.GL_FRAGMENT_SHADER,
                "#version 150\n" +
                "in vec2 vUV;\n" +
                "uniform sampler2D uAtlas;\n" +
                "void main() { if (texture(uAtlas, vUV).a < 0.5) discard; }\n");

            cprog = GL20.glCreateProgram();
            GL20.glAttachShader(cprog, cvs);
            GL20.glAttachShader(cprog, cfs);
            GL20.glBindAttribLocation(cprog, 0, "aPos");
            GL20.glBindAttribLocation(cprog, 1, "aUV");
            GL20.glLinkProgram(cprog);
            if (GL20.glGetProgrami(cprog, GL20.GL_LINK_STATUS) == GL11.GL_FALSE)
            {
                throw new IllegalStateException("cutout link: " + GL20.glGetProgramInfoLog(cprog));
            }
            GL20.glDeleteShader(cvs);
            cvs = 0;
            GL20.glDeleteShader(cfs);
            cfs = 0;

            newCutoutVao = GL30.glGenVertexArrays();

            // Resolve every uniform location before publishing any GL object to
            // the static fields. If a query fails, the catch below still owns and
            // deletes every temporary program and VAO.
            int newUViewProjLoc = GL20.glGetUniformLocation(prog, "uViewProj");
            int newCutoutViewProjLoc = GL20.glGetUniformLocation(cprog, "uViewProj");
            int newCutoutAtlasLoc = GL20.glGetUniformLocation(cprog, "uAtlas");

            program = prog;
            uViewProjLoc = newUViewProjLoc;
            vao = newVao;
            cutoutProgram = cprog;
            cutoutViewProjLoc = newCutoutViewProjLoc;
            cutoutAtlasLoc = newCutoutAtlasLoc;
            cutoutVao = newCutoutVao;

            prog = 0;
            newVao = 0;
            cprog = 0;
            newCutoutVao = 0;
        }
        catch (Throwable t)
        {
            if (vs != 0) { GL20.glDeleteShader(vs); }
            if (fs != 0) { GL20.glDeleteShader(fs); }
            if (cvs != 0) { GL20.glDeleteShader(cvs); }
            if (cfs != 0) { GL20.glDeleteShader(cfs); }
            if (prog != 0) { GL20.glDeleteProgram(prog); }
            if (cprog != 0) { GL20.glDeleteProgram(cprog); }
            if (newVao != 0) { GL30.glDeleteVertexArrays(newVao); }
            if (newCutoutVao != 0) { GL30.glDeleteVertexArrays(newCutoutVao); }
            glBroken = true;
            System.err.println("[irl-core] shadow depth program init failed: " + t);
            t.printStackTrace();
        }
    }

    private static int compile(int type, String src)
    {
        int sh = GL20.glCreateShader(type);
        GL20.glShaderSource(sh, src);
        GL20.glCompileShader(sh);
        if (GL20.glGetShaderi(sh, GL20.GL_COMPILE_STATUS) == GL11.GL_FALSE)
        {
            String log = GL20.glGetShaderInfoLog(sh);
            GL20.glDeleteShader(sh);
            throw new IllegalStateException("compile: " + log);
        }
        return sh;
    }

    /** Free one lamp's cached block VBO. */
    public static void releaseBlockVbo(long id)
    {
        int vbo = vboId.remove(id);
        if (vbo != 0)
        {
            GL15.glDeleteBuffers(vbo);
        }
        vboList.remove(id);
        vboVertCount.remove(id);
    }

    /** Free block VBOs (opaque + cutout) for lamps not in {@code liveIds} (gone,
     *  or feature off -> empty set drains all). Run once per bake after the light
     *  loops. A lamp with only cutout blocks isn't in the opaque {@link #vboList},
     *  so the cutout cache is swept separately. */
    public static void retainBlockVbos(LongSet liveIds)
    {
        if (!vboId.isEmpty())
        {
            ObjectIterator<Long2ObjectMap.Entry<List<BlockShadowEntry>>> it = vboList.long2ObjectEntrySet().iterator();
            while (it.hasNext())
            {
                long key = it.next().getLongKey();
                if (!liveIds.contains(key))
                {
                    int vbo = vboId.remove(key);
                    if (vbo != 0)
                    {
                        GL15.glDeleteBuffers(vbo);
                    }
                    vboVertCount.remove(key);
                    it.remove();
                }
            }
        }

        if (!cutoutVboId.isEmpty())
        {
            ObjectIterator<Long2ObjectMap.Entry<List<BlockShadowEntry>>> it = cutoutList.long2ObjectEntrySet().iterator();
            while (it.hasNext())
            {
                long key = it.next().getLongKey();
                if (!liveIds.contains(key))
                {
                    int vbo = cutoutVboId.remove(key);
                    if (vbo != 0)
                    {
                        GL15.glDeleteBuffers(vbo);
                    }
                    cutoutVertCount.remove(key);
                    it.remove();
                }
            }
        }
    }

    public static void endPass()
    {
        if (!inPass)
        {
            return;
        }

        try
        {
            // Draw the caster silhouettes before restoring state: they share this
            // pass's currentView/currentProj. Even if the draw fails, the outer
            // framebuffer/viewport/scissor state must still be restored.
            flushCasters();
        }
        finally
        {
            try
            {
                restoreGlStateStep(() -> GL11.glDepthMask(savedDepthMask));
                if (savedScissorEnabled)
                {
                    restoreGlStateStep(() -> GL11.glEnable(GL11.GL_SCISSOR_TEST));
                    restoreGlStateStep(() -> GL11.glScissor(savedScissorBox[0], savedScissorBox[1], savedScissorBox[2], savedScissorBox[3]));
                }
                else
                {
                    restoreGlStateStep(() -> GL11.glDisable(GL11.GL_SCISSOR_TEST));
                }
                restoreGlStateStep(() -> GL11.glViewport(savedViewport[0], savedViewport[1], savedViewport[2], savedViewport[3]));
                restoreGlStateStep(() -> GL30.glBindFramebuffer(GL30.GL_DRAW_FRAMEBUFFER, savedFbo));
            }
            finally
            {
                inPass = false;
            }
        }
    }

    /**
     * Abort the active shadow pass after an exception escapes the bake pipeline.
     * Restore the original caller state without flushing partially collected casters.
     */
    public static void abortCurrentPass()
    {
        abortPass();
    }

    /**
     * Abort a pass whose begin/setup failed before endPass.
     * Restore the original caller state without flushing partially collected casters.
     */
    private static void abortPass()
    {
        if (!inPass)
        {
            return;
        }

        try
        {
            restoreGlStateStep(() -> GL11.glDepthMask(savedDepthMask));
            if (savedScissorEnabled)
            {
                restoreGlStateStep(() -> GL11.glEnable(GL11.GL_SCISSOR_TEST));
                restoreGlStateStep(() -> GL11.glScissor(savedScissorBox[0], savedScissorBox[1], savedScissorBox[2], savedScissorBox[3]));
            }
            else
            {
                restoreGlStateStep(() -> GL11.glDisable(GL11.GL_SCISSOR_TEST));
            }
            restoreGlStateStep(() -> GL11.glViewport(savedViewport[0], savedViewport[1], savedViewport[2], savedViewport[3]));
            restoreGlStateStep(() -> GL30.glBindFramebuffer(GL30.GL_DRAW_FRAMEBUFFER, savedFbo));
        }
        finally
        {
            casterAccum.clear();
            ShadowBakeState.setBaking(false);
            inPass = false;
        }
    }

    /**
     * Snapshot the GL state every endPass restores, once per bake (the glGet* are
     * CPU<->GPU sync points; up to ~112 passes per bake would otherwise stall on
     * each). {@link #beginBake} re-arms it.
     */
    private static void savePassState()
    {
        if (!passStateSaved)
        {
            // Capture every value before marking a pass active. If a state query
            // fails, begin*() must not leave inPass true with an incomplete
            // snapshot that endPass() could later apply to the renderer.
            savedFbo = GL11.glGetInteger(GL30.GL_DRAW_FRAMEBUFFER_BINDING);
            GL11.glGetIntegerv(GL11.GL_VIEWPORT, savedViewport);
            savedScissorEnabled = GL11.glIsEnabled(GL11.GL_SCISSOR_TEST);
            if (savedScissorEnabled)
            {
                GL11.glGetIntegerv(GL11.GL_SCISSOR_BOX, savedScissorBox);
            }
            savedDepthMask = GL11.glGetBoolean(GL11.GL_DEPTH_WRITEMASK);

            passStateSaved = true;
        }

        inPass = true;
    }

    /** Shared up-vectors for the spot lookAt. lookAt only reads the components,
     *  so a single instance of each is safe (and avoids a per-pass allocation). */
    private static final Vector3f UP_Y = new Vector3f(0f, 1f, 0f);
    private static final Vector3f UP_Z = new Vector3f(0f, 0f, 1f);

    private static Vector3f pickStableUp(float dy)
    {
        return Math.abs(dy) > 0.99f ? UP_Z : UP_Y;
    }


}
