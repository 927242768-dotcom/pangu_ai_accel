# PDS 2022.2-SP6.4 的命令行 shell 不能可靠地跨进程恢复 Device Map 阶段。
# 为避免 Flow-0044/Flow-0013，本兼容入口重新执行完整可信流程。
source [file join [file dirname [info script]] build_kv_cache.tcl]
