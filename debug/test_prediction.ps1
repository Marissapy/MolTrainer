# Test Prediction and Model Info Features
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Testing Prediction Features" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Test 1: Train a simple model first
Write-Host "[TEST 1] Training a model for prediction testing..." -ForegroundColor Yellow
python -m moltrainer -i debug/test_data.csv -train -target activity -smiles smiles -model rf -auto_split 3way -o debug/pred_test_model -n_estimators 50 -cv 0 -no_cv | Out-Null

if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ Model trained successfully" -ForegroundColor Green
} else {
    Write-Host "  ✗ Model training failed" -ForegroundColor Red
    exit 1
}

# Test 2: Display model information
Write-Host "`n[TEST 2] Displaying model information..." -ForegroundColor Yellow
python -m moltrainer -model_info debug/pred_test_model/test_data_model.pkl

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n  ✓ Model info displayed successfully" -ForegroundColor Green
} else {
    Write-Host "`n  ✗ Model info display failed" -ForegroundColor Red
    exit 1
}

# Test 3: Make predictions on new data
Write-Host "`n[TEST 3] Making predictions on new data..." -ForegroundColor Yellow
python -m moltrainer -predict -load_model debug/pred_test_model/test_data_model.pkl -i debug/test_data.csv -o debug/predictions.csv

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n  ✓ Predictions generated successfully" -ForegroundColor Green
    
    # Check if prediction file exists
    if (Test-Path debug/predictions.csv) {
        Write-Host "  ✓ Prediction file created" -ForegroundColor Green
        
        # Show first few predictions
        Write-Host "`n  First 5 predictions:" -ForegroundColor Cyan
        Get-Content debug/predictions.csv | Select-Object -First 6
    } else {
        Write-Host "  ✗ Prediction file not found" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "`n  ✗ Prediction failed" -ForegroundColor Red
    exit 1
}

# Test 4: Test deep search
Write-Host "`n[TEST 4] Testing deep hyperparameter search..." -ForegroundColor Yellow
python -m moltrainer -i debug/test_data.csv -train -target activity -smiles smiles -model rf -search random -search_depth deep -search_iter 3 -search_cv 2 -auto_split 2way -o debug/deep_search_test -cv 0 -no_cv 2>&1 | Select-String -Pattern "search|Best|parameters" -Context 0,0

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n  ✓ Deep search completed successfully" -ForegroundColor Green
} else {
    Write-Host "`n  ✗ Deep search failed" -ForegroundColor Red
    exit 1
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "All Prediction Tests PASSED!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan

