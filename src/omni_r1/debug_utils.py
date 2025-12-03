import os

import torch


# 在函数内部检查是否已导入debugpy
def wait_for_debugger_if_rank0(port: int = 12427):
    """
    Only rank-0 process will start debugpy server for remote debugging.
    No-op if debugpy is not available or DEBUG_MODE is not set.

    Args:
        port (int): Port to listen on. Default is 7788.
    """
    # 检查是否应该启用调试
    if os.getenv("DEBUG_MODE") != "true":
        return

    # 尝试导入debugpy
    try:
        import debugpy
    except ImportError:
        print("debugpy not available, skipping debugger setup")
        return

    # 获取rank
    if not torch.distributed.is_initialized():
        rank = 0
    else:
        rank = torch.distributed.get_rank()

    # 只在rank 0进程设置调试器
    if rank == 0:
        print(f"[Debug Rank 0] pid: {os.getpid()}, listening on port {port}...")
        try:
            debugpy.listen(("0.0.0.0", port))
            print("[Debug Rank 0] Waiting for debugger attach...")
            debugpy.wait_for_client()
        except RuntimeError as e:
            print(f"[Debug Rank 0] Failed to listen on port {port}: {e}")
