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

Run-Step "1/9  Elo inicial anclado (eloratings.net)" `
    "python -m src.data.elo_loader"

Run-Step "2/9  Builder: fixtures + Elo + venue + distancia" `
    "python -m src.data.builder"

Run-Step "3/9  Referee rates (empirical-Bayes)" `
    "python -m src.features.referee_rates --referees data/raw/referees_wc2026.csv --output data/processed/referees_with_stats.csv"

Run-Step "4/9  Feature engineering" `
    "python src/features/feature_engineering.py"

Run-Step "5/9  XGBoost (todos los mercados)" `
    "python -m src.models.xgboost_model --features data/processed/features_train.csv --wc-features data/processed/features_wc2026.csv"

Run-Step "6/9  Poisson (goles)" `
    "python -m src.models.poisson_model --features data/processed/features_train.csv --wc-features data/processed/features_wc2026.csv"

Run-Step "7/9  Odds converter XGBoost" `
    "python -m src.evaluation.odds_converter --input outputs/wc2026_predictions.csv"

Run-Step "8/9  Odds converter Poisson" `
    "python -m src.evaluation.odds_converter --input outputs/wc2026_poisson_predictions.csv"

Run-Step "9/9  Model comparison" `
    "python -m src.evaluation.model_comparison"

Write-Host ""
Write-Host "Pipeline completo. Revisa outputs/model_comparison.csv" -ForegroundColor Yellow