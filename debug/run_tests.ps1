# MolTrainer Debug Test Suite
# Run comprehensive tests to validate all functionality

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "MolTrainer Debug Test Suite" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$ErrorCount = 0
$SuccessCount = 0

# Test 1: Descriptive Statistics
Write-Host "[TEST 1] Descriptive Statistics..." -ForegroundColor Yellow
try {
    python -m moltrainer -i debug/test_data.csv -desc_stats
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ PASSED" -ForegroundColor Green
        $SuccessCount++
    } else {
        Write-Host "  ✗ FAILED (Exit code: $LASTEXITCODE)" -ForegroundColor Red
        $ErrorCount++
    }
} catch {
    Write-Host "  ✗ ERROR: $_" -ForegroundColor Red
    $ErrorCount++
}

# Test 2: Data Cleaning (SMILES validation)
Write-Host "`n[TEST 2] Data Cleaning (SMILES validation)..." -ForegroundColor Yellow
try {
    python -m moltrainer -i debug/test_data.csv -clean -validate_smiles -smiles_column smiles -o debug/cleaned_data.csv
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ PASSED" -ForegroundColor Green
        $SuccessCount++
    } else {
        Write-Host "  ✗ FAILED (Exit code: $LASTEXITCODE)" -ForegroundColor Red
        $ErrorCount++
    }
} catch {
    Write-Host "  ✗ ERROR: $_" -ForegroundColor Red
    $ErrorCount++
}

# Test 3: Data Splitting
Write-Host "`n[TEST 3] Data Splitting (3-way stratified)..." -ForegroundColor Yellow
try {
    python -m moltrainer -i debug/cleaned_data.csv -split -stratify activity
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ PASSED" -ForegroundColor Green
        $SuccessCount++
    } else {
        Write-Host "  ✗ FAILED (Exit code: $LASTEXITCODE)" -ForegroundColor Red
        $ErrorCount++
    }
} catch {
    Write-Host "  ✗ ERROR: $_" -ForegroundColor Red
    $ErrorCount++
}

# Test 4: Classification Training (RF, 2-way split, config file)
Write-Host "`n[TEST 4] Classification Training (RF, 2-way auto-split, config)..." -ForegroundColor Yellow
try {
    python -m moltrainer -config debug/test_config_classification.yaml
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ PASSED" -ForegroundColor Green
        $SuccessCount++
    } else {
        Write-Host "  ✗ FAILED (Exit code: $LASTEXITCODE)" -ForegroundColor Red
        $ErrorCount++
    }
} catch {
    Write-Host "  ✗ ERROR: $_" -ForegroundColor Red
    $ErrorCount++
}

# Test 5: Regression Training (numeric features, 3-way split)
Write-Host "`n[TEST 5] Regression Training (numeric features, 3-way auto-split)..." -ForegroundColor Yellow
try {
    python -m moltrainer -config debug/test_config_regression.yaml
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ PASSED" -ForegroundColor Green
        $SuccessCount++
    } else {
        Write-Host "  ✗ FAILED (Exit code: $LASTEXITCODE)" -ForegroundColor Red
        $ErrorCount++
    }
} catch {
    Write-Host "  ✗ ERROR: $_" -ForegroundColor Red
    $ErrorCount++
}

# Test 6: SVM Training (command line)
Write-Host "`n[TEST 6] SVM Classification (command line)..." -ForegroundColor Yellow
try {
    python -m moltrainer -i debug/cleaned_data.csv -train -target activity -smiles smiles -model svm -auto_split none -val debug/cleaned_data_val.csv -test debug/cleaned_data_test.csv -o debug/results_svm -v
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ PASSED" -ForegroundColor Green
        $SuccessCount++
    } else {
        Write-Host "  ✗ FAILED (Exit code: $LASTEXITCODE)" -ForegroundColor Red
        $ErrorCount++
    }
} catch {
    Write-Host "  ✗ ERROR: $_" -ForegroundColor Red
    $ErrorCount++
}

# Test 7: Visualization
Write-Host "`n[TEST 7] Data Visualization..." -ForegroundColor Yellow
try {
    python -m moltrainer -i debug/cleaned_data.csv -visualize -plot_type distribution -columns "logp,molecular_weight" -sample_size 30 -o debug/vis_distribution.png
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ PASSED" -ForegroundColor Green
        $SuccessCount++
    } else {
        Write-Host "  ✗ FAILED (Exit code: $LASTEXITCODE)" -ForegroundColor Red
        $ErrorCount++
    }
} catch {
    Write-Host "  ✗ ERROR: $_" -ForegroundColor Red
    $ErrorCount++
}

# Test 8: Create Example Config
Write-Host "`n[TEST 8] Create Example Config..." -ForegroundColor Yellow
try {
    python -m moltrainer -create_config debug/example_config.yaml
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ PASSED" -ForegroundColor Green
        $SuccessCount++
    } else {
        Write-Host "  ✗ FAILED (Exit code: $LASTEXITCODE)" -ForegroundColor Red
        $ErrorCount++
    }
} catch {
    Write-Host "  ✗ ERROR: $_" -ForegroundColor Red
    $ErrorCount++
}

# Summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Test Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Passed: $SuccessCount" -ForegroundColor Green
Write-Host "Failed: $ErrorCount" -ForegroundColor Red
Write-Host "Total:  $($SuccessCount + $ErrorCount)" -ForegroundColor Cyan

if ($ErrorCount -eq 0) {
    Write-Host "`n✓ All tests PASSED!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "`n✗ Some tests FAILED!" -ForegroundColor Red
    exit 1
}

