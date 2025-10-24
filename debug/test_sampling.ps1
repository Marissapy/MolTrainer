# Test Data Sampling Features
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Testing Data Sampling Features" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$ErrorCount = 0
$SuccessCount = 0

# Test 1: Random sampling (absolute number)
Write-Host "[TEST 1] Random sampling (20 rows)..." -ForegroundColor Yellow
python -m moltrainer -i debug/test_data.csv -sample -n 20 -o debug/sample_random_20.csv | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ PASSED" -ForegroundColor Green
    $SuccessCount++
} else {
    Write-Host "  ✗ FAILED" -ForegroundColor Red
    $ErrorCount++
}

# Test 2: Random sampling (percentage)
Write-Host "`n[TEST 2] Random sampling (50%)..." -ForegroundColor Yellow
python -m moltrainer -i debug/test_data.csv -sample -n 50% -o debug/sample_random_50pct.csv | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ PASSED" -ForegroundColor Green
    $SuccessCount++
} else {
    Write-Host "  ✗ FAILED" -ForegroundColor Red
    $ErrorCount++
}

# Test 3: Stratified sampling
Write-Host "`n[TEST 3] Stratified sampling (60%, by activity)..." -ForegroundColor Yellow
python -m moltrainer -i debug/test_data.csv -sample -n 60% -sample_method stratified -stratify activity -o debug/sample_stratified.csv | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ PASSED" -ForegroundColor Green
    $SuccessCount++
} else {
    Write-Host "  ✗ FAILED" -ForegroundColor Red
    $ErrorCount++
}

# Test 4: Systematic sampling
Write-Host "`n[TEST 4] Systematic sampling (10 rows)..." -ForegroundColor Yellow
python -m moltrainer -i debug/test_data.csv -sample -n 10 -sample_method systematic -o debug/sample_systematic.csv | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ PASSED" -ForegroundColor Green
    $SuccessCount++
} else {
    Write-Host "  ✗ FAILED" -ForegroundColor Red
    $ErrorCount++
}

# Test 5: Head sampling
Write-Host "`n[TEST 5] Head sampling (first 15 rows)..." -ForegroundColor Yellow
python -m moltrainer -i debug/test_data.csv -sample -n 15 -sample_method head -o debug/sample_head.csv | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ PASSED" -ForegroundColor Green
    $SuccessCount++
} else {
    Write-Host "  ✗ FAILED" -ForegroundColor Red
    $ErrorCount++
}

# Test 6: Tail sampling
Write-Host "`n[TEST 6] Tail sampling (last 15 rows)..." -ForegroundColor Yellow
python -m moltrainer -i debug/test_data.csv -sample -n 15 -sample_method tail -o debug/sample_tail.csv | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ PASSED" -ForegroundColor Green
    $SuccessCount++
} else {
    Write-Host "  ✗ FAILED" -ForegroundColor Red
    $ErrorCount++
}

# Test 7: Sampling with replacement
Write-Host "`n[TEST 7] Sampling with replacement (30 rows)..." -ForegroundColor Yellow
python -m moltrainer -i debug/test_data.csv -sample -n 30 -replace -o debug/sample_replace.csv | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ PASSED" -ForegroundColor Green
    $SuccessCount++
} else {
    Write-Host "  ✗ FAILED" -ForegroundColor Red
    $ErrorCount++
}

# Summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Sampling Test Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Passed: $SuccessCount" -ForegroundColor Green
Write-Host "Failed: $ErrorCount" -ForegroundColor Red
Write-Host "Total:  $($SuccessCount + $ErrorCount)" -ForegroundColor Cyan

if ($ErrorCount -eq 0) {
    Write-Host "`n✓ All sampling tests PASSED!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "`n✗ Some sampling tests FAILED!" -ForegroundColor Red
    exit 1
}

