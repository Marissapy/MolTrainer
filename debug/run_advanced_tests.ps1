# MolTrainer Advanced Debug Test Suite
# Test edge cases and advanced features

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "MolTrainer Advanced Debug Test Suite" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$ErrorCount = 0
$SuccessCount = 0

# Test 1: Data Cleaning - Comprehensive (dirty data)
Write-Host "[ADVANCED TEST 1] Comprehensive Data Cleaning..." -ForegroundColor Yellow
try {
    python -m moltrainer -i debug/test_data_dirty.csv -clean -remove_duplicates -handle_missing -missing_method drop -validate_smiles -smiles_column smiles -filter_value "ic50 < 200" -o debug/cleaned_dirty_data.csv
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

# Test 2: Data Cleaning - Outlier Removal (IQR method)
Write-Host "`n[ADVANCED TEST 2] Outlier Removal (IQR)..." -ForegroundColor Yellow
try {
    python -m moltrainer -i debug/test_data.csv -clean -remove_outliers -outlier_method iqr -outlier_columns ic50 -outlier_threshold 1.5 -o debug/no_outliers_iqr.csv
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

# Test 3: Data Cleaning - Outlier Removal (Z-score method)
Write-Host "`n[ADVANCED TEST 3] Outlier Removal (Z-score)..." -ForegroundColor Yellow
try {
    python -m moltrainer -i debug/test_data.csv -clean -remove_outliers -outlier_method zscore -outlier_columns ic50,logp -outlier_threshold 2.5 -o debug/no_outliers_zscore.csv
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

# Test 4: Visualization - Correlation Heatmap
Write-Host "`n[ADVANCED TEST 4] Correlation Heatmap..." -ForegroundColor Yellow
try {
    python -m moltrainer -i debug/test_data.csv -visualize -plot_type correlation -columns "ic50,logp,molecular_weight" -o debug/correlation.svg
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

# Test 5: Visualization - Boxplot with sampling
Write-Host "`n[ADVANCED TEST 5] Boxplot with 60% sampling..." -ForegroundColor Yellow
try {
    python -m moltrainer -i debug/test_data.csv -visualize -plot_type boxplot -columns "ic50,logp" -sample_size 60% -title "IC50 and LogP Distribution" -o debug/boxplot.jpg
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

# Test 6: Data Splitting - 2-way split with custom ratios
Write-Host "`n[ADVANCED TEST 6] 2-way Split (80/20)..." -ForegroundColor Yellow
try {
    python -m moltrainer -i debug/test_data.csv -split -train_ratio 0.8 -val_ratio 0.0 -test_ratio 0.2 -shuffle
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

# Test 7: Training - Hyperparameter Search (Random Search with XGBoost)
Write-Host "`n[ADVANCED TEST 7] Hyperparameter Search (Random, XGBoost)..." -ForegroundColor Yellow
try {
    python -m moltrainer -config debug/test_advanced.yaml
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

# Test 8: Training - LightGBM with Grid Search
Write-Host "`n[ADVANCED TEST 8] Grid Search (LightGBM)..." -ForegroundColor Yellow
try {
    python -m moltrainer -i debug/test_data.csv -train -target activity -smiles smiles -model lgb -search grid -search_cv 3 -auto_split 2way -o debug/results_lgb_grid -cv 0 -no_cv
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

# Test 9: Training - Logistic Regression (Linear Model)
Write-Host "`n[ADVANCED TEST 9] Logistic Regression..." -ForegroundColor Yellow
try {
    python -m moltrainer -i debug/test_data.csv -train -target activity -features "logp,molecular_weight" -model lr -task classification -auto_split 3way -o debug/results_lr
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

# Test 10: Training - Linear Regression (Regression Task)
Write-Host "`n[ADVANCED TEST 10] Linear Regression (Ridge)..." -ForegroundColor Yellow
try {
    # First split the data to create val and test files
    python -m moltrainer -i debug/test_data.csv -split -train_ratio 0.7 -val_ratio 0.15 -test_ratio 0.15 | Out-Null
    # Then train using the split data
    python -m moltrainer -i debug/test_data_train.csv -train -target ic50 -features "logp,molecular_weight" -model lr -task regression -auto_split none -val debug/test_data_val.csv -test debug/test_data_test.csv -o debug/results_ridge
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
Write-Host "Advanced Test Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Passed: $SuccessCount" -ForegroundColor Green
Write-Host "Failed: $ErrorCount" -ForegroundColor Red
Write-Host "Total:  $($SuccessCount + $ErrorCount)" -ForegroundColor Cyan

if ($ErrorCount -eq 0) {
    Write-Host "`n✓ All advanced tests PASSED!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "`n✗ Some advanced tests FAILED!" -ForegroundColor Red
    exit 1
}

