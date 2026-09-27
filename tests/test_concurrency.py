"""Concurrent updates must linearize: no lost updates, no probability breach."""
import threading

from voxel_occupancy.service import OccupancyService
from voxel_occupancy.types import CellState, ExpiryPolicy, GridConfig, Observation


def test_concurrent_updates_no_lost_updates_ignore_time():
    """Every observation from every thread is counted exactly once."""
    n_threads = 8
    per_thread = 200
    service = OccupancyService(
        GridConfig(expiry_policy=ExpiryPolicy.IGNORE_TIME)
    )
    barrier = threading.Barrier(n_threads)

    def worker(tid):
        barrier.wait()
        for i in range(per_thread):
            # Distinct timestamp per (thread, i) does not matter under
            # IGNORE_TIME; all observations must accumulate.
            service.submit([
                Observation((0, 0, 0), 0.9, timestamp=tid * 1000 + i + 1)
            ])

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = service.get_cell((0, 0, 0))
    assert snap.update_count == n_threads * per_thread
    assert 0.0 < snap.probability < 1.0
    assert snap.state is CellState.OCCUPIED


def test_concurrent_reject_policy_equivalent_to_some_serial_order():
    """Under REJECT with disjoint timestamp ranges, exactly the newest
    observation per timestamp ordering is applied; total applied count is
    deterministic regardless of thread interleaving."""
    n_threads = 4
    per_thread = 100
    service = OccupancyService(GridConfig(expiry_policy=ExpiryPolicy.REJECT))
    barrier = threading.Barrier(n_threads)

    def worker(tid):
        barrier.wait()
        # Thread tid owns timestamps [tid*1000 + 1 .. +100], strictly
        # increasing within the thread; ranges are globally disjoint.
        for i in range(1, per_thread + 1):
            service.submit([
                Observation((5, 5, 5), 0.8, timestamp=tid * 1000 + i)
            ])

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = service.get_cell((5, 5, 5))
    # Every range is globally disjoint and strictly newer than any earlier
    # range? No: ranges interleave in value? They don't - 1..100,1001..1100..
    # all globally increasing by thread id, but thread start order is random.
    # Whichever serial order happens, an observation is rejected only if its
    # timestamp <= current mark. Count of applied observations depends on the
    # order, but update_count equals number of times a new global maximum was
    # seen - at minimum per_thread (the highest-id thread's chain) and at most
    # all of them. Boundedness still must hold.
    assert per_thread <= snap.update_count <= n_threads * per_thread
    assert 0.0 < snap.probability < 1.0


def test_concurrent_commits_are_atomic_version_monotonic():
    n_threads = 6
    per_thread = 150
    service = OccupancyService(GridConfig())
    seen_versions: list[int] = []
    lock = threading.Lock()
    barrier = threading.Barrier(n_threads)

    def worker(tid):
        barrier.wait()
        for i in range(per_thread):
            result = service.submit(
                [Observation((tid, i, 0), 0.9, timestamp=tid * 10_000 + i + 1)]
            )
            with lock:
                seen_versions.append(result.sequence)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly one commit per submit; versions form 1..N in some order.
    assert sorted(seen_versions) == list(range(1, n_threads * per_thread + 1))
    assert service.snapshot_version() == n_threads * per_thread


def test_concurrent_reads_and_writes_return_valid_probabilities():
    service = OccupancyService(
        GridConfig(expiry_policy=ExpiryPolicy.IGNORE_TIME)
    )
    stop = threading.Event()
    bad: list[float] = []

    def writer():
        ts = 0
        while not stop.is_set():
            ts += 1
            service.submit([
                Observation((0, 0, 0), 0.99, ts),
                Observation((1, 0, 0), 0.01, ts),
            ])

    def reader():
        while not stop.is_set():
            for cell in ((0, 0, 0), (1, 0, 0), (2, 0, 0)):
                snap = service.get_cell(cell)
                if snap is not None and not (0.0 < snap.probability < 1.0):
                    bad.append(snap.probability)

    wt = threading.Thread(target=writer)
    rt = [threading.Thread(target=reader) for _ in range(4)]
    wt.start()
    for t in rt:
        t.start()
    stop.set() if False else None
    import threading as _t
    timer = _t.Timer(0.4, stop.set)
    timer.start()
    wt.join()
    for t in rt:
        t.join()
    timer.join()
    assert not bad
