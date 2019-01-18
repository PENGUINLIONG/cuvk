function CollectCuvk {
    param([int] $i)
    function CaptureTime {
        param ([int] $j, [string] $task)
        $times =
            Get-Content "bench_cuvk.$($i)bac.$j.log" |
            Select-String $task |
            ForEach-Object { return [float](([string]$_).Substring(0, 12)) };
        return $times[1] - $times[0];
    }
    $deform_list = @()
    $eval_list = @()
    for ($j = 0; $j -lt 10; ++$j) {
        $deform_list += CaptureTime $j "deformation";
        $eval_list += CaptureTime $j "evaluation";
    }
    return @(($deform_list | Measure-Object -Average).Average, ($eval_list | Measure-Object -Average).Average)
}
function CollectCubl {
    param([int] $i)
    function CaptureTime {
        param ([int] $j, [string] $task)
        $time =
            Get-Content "bench_cubl.$($i)bac.$j.log" |
            Select-String $task;
        return [float]($time -split ' ')[3]
    }
    $deform_list = @()
    $eval_list = @()
    for ($j = 0; $j -lt 10; ++$j) {
        $deform_list += CaptureTime $j "deformation";
        $eval_list += CaptureTime $j "evaluation";
    }
    return @(($deform_list | Measure-Object -Average).Average, ($eval_list | Measure-Object -Average).Average)
}




$cuvk_deform_list = @()
$cuvk_eval_list = @()
$cubl_deform_list = @()
$cubl_eval_list = @()

Remove-Item "*.log"

for ($i = 10; $i -le 100; $i += 10) {
    $env:L_BAC_COUNT = $i;

    for ($j = 0; $j -lt 10; $j++) {
        $env:L_BENCH_TYPE = 'CUVK';
        $file = "bench_cuvk.$($i)bac.$j.log";
        python python/demo.py 2>&1 | %{ "$_" } | Out-File $file -NoNewline;

        $env:L_BENCH_TYPE = 'CUBL';
        $file = "bench_cubl.$($i)bac.$j.log";
        python python/demo.py 2>&1 | %{ "$_" } | Out-File $file -NoNewline;
    }

    $cuvk_deform,$cuvk_eval = CollectCuvk($i)
    $cubl_deform,$cubl_eval = CollectCubl($i)

    $cuvk_deform_list += $cuvk_deform
    $cuvk_eval_list += $cuvk_eval
    $cubl_deform_list += $cubl_deform
    $cubl_eval_list += $cubl_eval
    
    Write-Host "collected data for $i bacteria: $cuvk_deform, $cuvk_eval, $cubl_deform, $cubl_eval"
}


$cuvk_deform_list = $cuvk_deform_list -join ','
$cuvk_eval_list = $cuvk_eval_list -join ','
$cubl_deform_list = $cubl_deform_list -join ','
$cubl_eval_list = $cubl_eval_list -join ','

$py = 
    "import matplotlib.pyplot as plot`n" +
    "import numpy as np`n" +
    "plot.plot([10 * (i + 1) for i in range(10)], [$cuvk_deform_list])`n" +
    "plot.show()`n" +
    "plot.plot([200 * (i + 1) for i in range(10)], [$cuvk_eval_list])`n" +
    "plot.show()`n" +
    "plot.plot([10 * i for i in range(10)], [$cubl_deform_list])`n" +
    "plot.show()`n" +
    "plot.plot([200 * i for i in range(10)], [$cubl_eval_list])`n" +
    "plot.show()";

$py | python
