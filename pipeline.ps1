# run_pipeline.ps1
# ================
# Pipeline completo de prediccion WC2026.
# Corta al primer fallo. Ejecutar desde la raiz del repo:
#     powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1
#
# Empieza en builder porque los pasos de convocatorias/ratings no cambian con
# el bump de Elo a USA (solo se toco builder.py). Si cambias squads o ratings,
# usa regen_fifa.ps1 (que reejecuta tambien esos pasos previos).
#
# Pasos:
#   builder -> feature_engineering -> XGBoost -> Poisson -> cuotas (x2)
#   -> model_comparison -> model_ensemble
#   -> calibration_report -> match_reliability -> ev_calculator
#
# ev_calculator necesita ODDS_API_KEY en .env y conexion a internet.

$ErrorActionPreference = "Stop"

function Run-Step {
    param([string]$Description, [string[]]$CommandArgs)
    Write-Host ""
    Write-Host ">> $Description" -ForegroundColor Cyan
    & python @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FALLO] en: $Description (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# --- Datos y features ---
Run-Step "Builder dataset enriquecido"      @("-m", "src.data.builder")
Run-Step "Feature engineering"              @("src/features/feature_engineering.py")

# --- Modelos ---
Run-Step "XGBoost (5 mercados)"             @("-m", "src.models.xgboost_model", "--markets", "result_1x2,over25,btts,total_goals,corners")
Run-Step "Modelo Poisson"                   @("-m", "src.models.poisson_model")

# --- Cuotas y combinacion ---
Run-Step "Cuotas XGBoost"                   @("-m", "src.evaluation.odds_converter", "--input", "outputs/wc2026_predictions.csv")
Run-Step "Cuotas Poisson"                   @("-m", "src.evaluation.odds_converter", "--input", "outputs/wc2026_poisson_predictions.csv")
Run-Step "Comparacion de modelos"           @("-m", "src.evaluation.model_comparison")
Run-Step "Ensemble final"                   @("-m", "src.evaluation.model_ensemble")

# --- Evaluacion ---
Run-Step "Calibracion (historico)"          @("-m", "src.evaluation.calibration_report")
Run-Step "Fiabilidad de inputs por partido" @("-m", "src.evaluation.match_reliability")
Run-Step "Calculo de EV (cuotas reales)"    @("-m", "src.evaluation.ev_calculator")

Write-Host ""
Write-Host "[OK] Pipeline completo sin errores." -ForegroundColor Green
Write-Host "Revisa: outputs/wc2026_ev.csv  (value bets con fiabilidad)" -ForegroundColor Green