$ErrorActionPreference = "Stop"

function Run-Step {
    param([string]$Label, [string]$Cmd)
    Write-Host ""
    Write-Host ">>> $Label" -ForegroundColor Cyan
    Invoke-Expression $Cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FALLO en: $Label (exit $LASTEXITCODE)" -ForegroundColor Red
        exit 1
    }
    Write-Host "OK" -ForegroundColor Green
}

# Regenera unified.csv con los 314 partidos. Re-descarga StatsBomb,
# asi que tarda unos minutos. Es el paso que mete Copa y AFCON en el dataset.
Run-Step "1/5  Recoleccion + unificacion (lento, re-descarga StatsBomb)" `
    "python data_collector.py"

Run-Step "2/5  Elo inicial anclado" `
    "python -m src.data.elo_loader"

Run-Step "3/5  Builder: regenera matches_enriched.csv (de aqui lee features)" `
    "python -m src.data.builder"

Run-Step "4/5  Feature engineering (con los 314)" `
    "python src/features/feature_engineering.py"

Run-Step "5/5  XGBoost (incluye el mercado de corners)" `
    "python -m src.models.xgboost_model --features data/processed/features_train.csv --wc-features data/processed/features_wc2026.csv"

Write-Host ""
Write-Host "Listo. Abre outputs/wc2026_predictions.csv y mira la columna de corners." -ForegroundColor Yellow
Write-Host "Antes predecia ~8; con los 314 deberia subir hacia ~10." -ForegroundColor Yellow