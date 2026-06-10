# run_all.ps1
# ===========
# Pipeline completo WC2026: datos -> features -> modelos -> ensemble
# -> evaluacion -> EV con cuotas reales (Winamax) -> shortlist + tracker
# -> radar multi-mercado (sin API).
#
# Corta al primer fallo. Ejecutar desde la raiz del repo:
#     powershell -ExecutionPolicy Bypass -File .\run_all.ps1
#
# Flags:
#     -SkipApi    No llama a The Odds API. Reutiliza outputs/wc2026_ev_h2h_strategy.csv
#                 si existe (no gasta creditos). Util para re-runs el mismo dia.
#     -Collect    Reejecuta data_collector.py (Kaggle + StatsBomb). Lento; solo
#                 si han cambiado los datos historicos.
#
# NO reejecuta squads/ratings/player_stats (usar regen_fifa.ps1 si cambian).
# El paso de EV usa winamax_fr (la casa donde se apuesta), no pinnacle.

param(
    [switch]$SkipApi,
    [switch]$Collect
)

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

# --- 0. Datos historicos (opcional, lento) ---
if ($Collect) {
    Run-Step "Recoleccion de datos (Kaggle + StatsBomb)" @("data_collector.py")
}

# --- 1. Arbitros: refrescar tasas modelo (shrinkage, idempotente) ---
Run-Step "Tasas de arbitro (shrinkage)" @("-m", "src.features.referee_rates", "--referees", "data/processed/referees_with_stats.csv", "--output", "data/processed/referees_with_stats.csv")

# --- 2. Dataset y features ---
Run-Step "Builder dataset enriquecido"      @("-m", "src.data.builder")
Run-Step "Feature engineering"              @("src/features/feature_engineering.py")

# --- 3. Modelos (6 mercados, incluye yellows analitico) ---
Run-Step "XGBoost (6 mercados)"             @("-m", "src.models.xgboost_model", "--markets", "result_1x2,over25,btts,total_goals,corners,yellows")
Run-Step "Modelo Poisson"                   @("-m", "src.models.poisson_model")

# --- 4. Cuotas del modelo y combinacion ---
Run-Step "Cuotas XGBoost"                   @("-m", "src.evaluation.odds_converter", "--input", "outputs/wc2026_predictions.csv")
Run-Step "Cuotas Poisson"                   @("-m", "src.evaluation.odds_converter", "--input", "outputs/wc2026_poisson_predictions.csv")
Run-Step "Comparacion de modelos"           @("-m", "src.evaluation.model_comparison")
Run-Step "Ensemble final"                   @("-m", "src.evaluation.model_ensemble")

# --- 5. Evaluacion local (sin API) ---
Run-Step "Calibracion (historico)"          @("-m", "src.evaluation.calibration_report")
Run-Step "Fiabilidad de inputs por partido" @("-m", "src.evaluation.match_reliability")

# --- 6. EV real + shortlist + tracker (1X2 v1, Winamax) ---
if ($SkipApi) {
    Run-Step "1X2 v1 (sin API, snapshot existente)" @("-m", "src.evaluation.run_1x2_v1", "--skip-ev", "--bookmakers", "winamax_fr")
} else {
    Run-Step "1X2 v1 (EV Winamax + shortlist + tracker)" @("-m", "src.evaluation.run_1x2_v1", "--bookmakers", "winamax_fr")
}

# --- 7. Cuotas flat de TODOS los mercados disponibles en Winamax ---
if (-not $SkipApi) {
    Run-Step "Cuotas flat multi-mercado (Winamax)" @("-m", "src.data.historical_odds_collector", "--current-flat", "--sport", "soccer_fifa_world_cup", "--markets", "h2h,totals,btts", "--bookmakers", "winamax_fr", "--regions", "eu", "--output", "data/processed/odds_current_worldcup_flat.csv")
    $flat = "data/processed/odds_current_worldcup_flat.csv"
    if ((Test-Path $flat) -and ((Get-Item $flat).Length -lt 10)) {
        Remove-Item $flat
        Write-Host "[INFO] Winamax sin totals/btts en la API todavia: snapshot vacio eliminado. El radar seguira solo con probabilidades del modelo." -ForegroundColor Yellow
    }
}

# --- 8. Radar multi-mercado: EV de todos los mercados + paper shortlist ---
Run-Step "Radar multi-mercado + EV + paper"  @("-m", "src.evaluation.run_market_radar")

# --- 9. Subir senales al tracker web ---
Run-Step "Subir senales a Supabase (tracker)" @("-m", "src.evaluation.push_signals_supabase")

Write-Host ""
Write-Host "[OK] Pipeline completo sin errores." -ForegroundColor Green
Write-Host ""
Write-Host "Resultados finales:" -ForegroundColor Green
Write-Host "  outputs/wc2026_ensemble_predictions.csv   (prediccion final por partido)"
Write-Host "  outputs/wc2026_ev_h2h_shortlist.csv       (apuestas core 1X2)"
Write-Host "  outputs/wc2026_ev_h2h_manual_review.csv   (revision manual / paper)"
Write-Host "  data/tracking/wc2026_bet_tracker.csv      (tracker con stakes y CLV)"
Write-Host "  outputs/wc2026_market_probabilities.csv   (radar multi-mercado)"
Write-Host "  outputs/wc2026_market_availability.csv    (estado de cada mercado)"
Write-Host "  outputs/wc2026_multi_market_ev.csv        (EV de TODOS los mercados con cuota)"
Write-Host "  outputs/wc2026_multi_market_paper_shortlist.csv (totals/btts filtrados, paper)"
Write-Host "  data/tracking/wc2026_paper_tracker.csv    (seguimiento paper con CLV)"
