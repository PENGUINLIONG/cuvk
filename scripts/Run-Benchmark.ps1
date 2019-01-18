$PSDefaultParameterValues = @{"Out-File:Encoding"="utf8"}
Remove-Item "*.log"
for ($i = 0; $i -lt 10; $i++) {
    $env:L_BENCH_TYPE = 'CUVK';
    python python/demo.py 2>&1 | %{ "$_" } | Out-File "bench_cuvk.$i.log" -NoNewline

    $env:L_BENCH_TYPE = 'CUBL';
    python python/demo.py 2>&1 | %{ "$_" } | Out-File "bench_cubl.$i.log" -NoNewline
}
