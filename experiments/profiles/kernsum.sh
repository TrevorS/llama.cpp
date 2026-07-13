#!/usr/bin/env zsh
# kernsum.sh <trace.nsys-rep> [window_seconds]
# Kernel time ranking within the LAST <window_seconds> of GPU activity
# (default: whole trace). Use the window to isolate the final measured
# llama-bench pass from the depth-fill ubatches that precede it.
#
# Typical use after a depth leg:
#   nsys profile -o lid_pp_dN --trace=cuda --sample=none --cpuctxsw=none \
#       <llama-bench ... -d N -r 1 -o json>
#   ./kernsum.sh lid_pp_dN.nsys-rep 11.3     # 11.3 = avg_ns of the measured pass
set -e
rep=$1
win=${2:-0}
sq=${rep%.nsys-rep}.sqlite
[[ $sq -nt $rep ]] || nsys export --type sqlite --force-overwrite=true -o $sq $rep >/dev/null 2>&1
if [[ $win == 0 ]]; then
    filter=""
else
    filter="WHERE start >= (SELECT MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL) - CAST(${win}*1e9 AS INTEGER)"
fi
sqlite3 -column $sq "
SELECT printf('%8.1f', SUM(k.end-k.start)/1e6) AS ms,
       printf('%5.1f%%', 100.0*SUM(k.end-k.start) /
           (SELECT SUM(k2.end-k2.start) FROM CUPTI_ACTIVITY_KIND_KERNEL k2 $filter)) AS share,
       COUNT(*) AS n,
       printf('%8.3f', AVG(k.end-k.start)/1e6) AS avg_ms,
       substr(s.value,1,70) AS kernel
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName = s.id
$filter
GROUP BY s.value ORDER BY SUM(k.end-k.start) DESC LIMIT 20;
SELECT printf('TOTAL %.1f ms GPU-busy in window', SUM(k.end-k.start)/1e6)
FROM CUPTI_ACTIVITY_KIND_KERNEL k $filter;"
