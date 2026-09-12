import Metal
let device = MTLCreateSystemDefaultDevice()!
print(device.name, "maxThreadsPerThreadgroup", device.maxThreadsPerThreadgroup, "maxTG mem", device.maxThreadgroupMemoryLength)
let tests: [(String, String)] = [
 ("relaxed fetch_add", """
 #include <metal_stdlib>
 using namespace metal;
 kernel void k(device atomic_uint *c [[buffer(0)]]) { atomic_fetch_add_explicit(c, 1u, memory_order_relaxed); }
 """),
 ("acq_rel fetch_add", """
 #include <metal_stdlib>
 using namespace metal;
 kernel void k(device atomic_uint *c [[buffer(0)]]) { atomic_fetch_add_explicit(c, 1u, memory_order_acq_rel); }
 """),
 ("acquire load", """
 #include <metal_stdlib>
 using namespace metal;
 kernel void k(device atomic_uint *c [[buffer(0)]], device uint *o [[buffer(1)]]) { o[0] = atomic_load_explicit(c, memory_order_acquire); }
 """),
 ("atomic_thread_fence device", """
 #include <metal_stdlib>
 using namespace metal;
 kernel void k(device atomic_uint *c [[buffer(0)]], device uint *o [[buffer(1)]]) {
   o[1] = 5u;
   atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
   atomic_fetch_add_explicit(c, 1u, memory_order_relaxed);
 }
 """),
 ("atomic_thread_fence 2-arg", """
 #include <metal_stdlib>
 using namespace metal;
 kernel void k(device atomic_uint *c [[buffer(0)]], device uint *o [[buffer(1)]]) {
   o[1] = 5u;
   atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);
   atomic_fetch_add_explicit(c, 1u, memory_order_relaxed);
 }
 """),
 ("volatile device load", """
 #include <metal_stdlib>
 using namespace metal;
 kernel void k(volatile device uint *c [[buffer(0)]], device uint *o [[buffer(1)]]) { o[0] = c[3]; }
 """),
 ("threadgroup_barrier mem_device", """
 #include <metal_stdlib>
 using namespace metal;
 kernel void k(device uint *o [[buffer(1)]]) { o[0]=1u; threadgroup_barrier(mem_flags::mem_device); }
 """),
]
for (name, src) in tests {
    do {
        let opts = MTLCompileOptions()
        _ = try device.makeLibrary(source: src, options: opts)
        print("OK   ", name)
    } catch {
        let msg = "\(error)".split(separator: "\n").filter { $0.contains("error") }.prefix(2).joined(separator: " | ")
        print("FAIL ", name, "::", msg.prefix(220))
    }
}
