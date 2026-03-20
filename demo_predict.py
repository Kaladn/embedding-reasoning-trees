"""
6-1-6 Prediction Demo — the sharp, demoable core.

Usage:
    python demo_predict.py
    python demo_predict.py --anchor ai
    python demo_predict.py --anchor security --chain 8
"""
import sys, io, time, argparse

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from cascade_tokenizer.predict import Predictor

STORE_PATH = "reasoning_cache/evidence.bin"


def demo(anchor: str = "ai", chain_depth: int = 6, k: int = 5):
    print(f"Loading store...")
    t0 = time.time()
    p = Predictor(STORE_PATH)
    print(f"Loaded {p.stats()['cells']:,} cells in {time.time()-t0:.2f}s\n")

    # Predict next
    print(f"[{anchor}] predict_next (top {k}):")
    t0 = time.time()
    nexts = p.predict_next(anchor, k=k)
    qtime = (time.time() - t0) * 1000
    for word, count in nexts:
        print(f"  +1  {word:25s} count={count}")
    print(f"  ({qtime:.1f}ms)\n")

    # Predict previous
    print(f"[{anchor}] predict_previous (top {k}):")
    prevs = p.predict_previous(anchor, k=k)
    for word, count in prevs:
        print(f"  -1  {word:25s} count={count}")

    # Forward chain
    print(f"\nForward chain ({chain_depth} steps):")
    chain = p.forward_chain(anchor, steps=chain_depth)
    words = [anchor] + [w for w, _ in chain]
    print(f"  {' -> '.join(words)}")

    # Backward chain
    print(f"\nBackward chain ({chain_depth} steps):")
    chain = p.backward_chain(anchor, steps=chain_depth)
    words = [w for w, _ in reversed(chain)] + [anchor]
    print(f"  {' -> '.join(words)}")

    # Full context
    print(f"\nFull 6-1-6 context:")
    ctx = p.full_context(anchor, k=k)
    for label in sorted(ctx.keys()):
        neighbors = ctx[label]
        items = ", ".join(f"{w}({c})" for w, c in neighbors)
        print(f"  {label:12s}: {items}")

    p.close()


def benchmark():
    print("=== BENCHMARK ===\n")

    # Cold load
    t0 = time.time()
    p = Predictor(STORE_PATH)
    cold = time.time() - t0
    print(f"Cold load:  {cold:.3f}s  ({p.stats()['cells']:,} cells)")

    # Single queries
    anchors = ["ai", "model", "data", "security", "training",
               "network", "python", "server", "encryption", "memory"]

    times = []
    for anchor in anchors:
        t0 = time.time()
        p.predict_next(anchor, k=10)
        times.append((time.time() - t0) * 1000)

    print(f"Query (10x): min={min(times):.1f}ms  max={max(times):.1f}ms  avg={sum(times)/len(times):.1f}ms")

    # Chain depth
    for depth in [1, 3, 6, 12]:
        t0 = time.time()
        for anchor in anchors:
            p.forward_chain(anchor, steps=depth)
        elapsed = (time.time() - t0) * 1000
        print(f"Chain d={depth:2d} (10x): {elapsed:.1f}ms total  {elapsed/10:.1f}ms/chain")

    # Batch 100 queries
    t0 = time.time()
    for _ in range(10):
        for anchor in anchors:
            p.predict_next(anchor, k=5)
    batch = (time.time() - t0) * 1000
    print(f"Batch 100q:  {batch:.1f}ms total  {batch/100:.1f}ms/query")

    p.close()


def main():
    parser = argparse.ArgumentParser(description="6-1-6 Prediction Demo")
    parser.add_argument("--anchor", default="ai", help="Anchor word to explore")
    parser.add_argument("--chain", type=int, default=6, help="Chain depth")
    parser.add_argument("--k", type=int, default=5, help="Top-K neighbors")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    args = parser.parse_args()

    if args.benchmark:
        benchmark()
    else:
        demo(args.anchor, args.chain, args.k)


if __name__ == "__main__":
    main()
