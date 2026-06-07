# regen_fifa.ps1
# ==============
# Regenera el pipeline tras adoptar el squad oficial de FIFA.
# Corta al primer fallo. Ejecutar desde la raiz del repo:
#     .\regen_fifa.ps1
#
# Asume que ya existen team_ratings.csv y unified.csv (no reejecuta
# elo_loader ni data_collector porque no dependen del squad).
# match_transfermarkt se omite: requiere players.csv.gz (no presente) y
# no es necesario para ratings/stats (leen sus propios CSV de data/transfermarkt).

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

Run-Step "Normalizar convocatorias"         @("-m", "src.data.normalize_squads")
Run-Step "Player master"                    @("-m", "src.data.player_master")
Run-Step "Rating compuesto WC2026"          @("-m", "src.data.team_rating_wc2026")
Run-Step "Stats de jugadores por seleccion" @("-m", "src.data.player_stats_aggregator")
Run-Step "Builder dataset enriquecido"      @("-m", "src.data.builder")
Run-Step "Feature engineering"              @("src/features/feature_engineering.py")
Run-Step "XGBoost (5 mercados)"             @("-m", "src.models.xgboost_model", "--markets", "result_1x2,over25,btts,total_goals,corners")
Run-Step "Modelo Poisson"                   @("-m", "src.models.poisson_model")
Run-Step "Cuotas XGBoost"                   @("-m", "src.evaluation.odds_converter", "--input", "outputs/wc2026_predictions.csv")
Run-Step "Cuotas Poisson"                   @("-m", "src.evaluation.odds_converter", "--input", "outputs/wc2026_poisson_predictions.csv")
Run-Step "Comparacion de modelos"           @("-m", "src.evaluation.model_comparison")
Run-Step "Ensemble final"                   @("-m", "src.evaluation.model_ensemble")

Write-Host ""
Write-Host "[OK] Pipeline completo sin errores." -ForegroundColor Green