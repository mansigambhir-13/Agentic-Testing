# PowerShell script to run reliability tests for MultiAgent V2
# Usage: .\run_reliability_tests.ps1 [quick|all|infrastructure|agent|recovery]

param(
    [string]$Type = "quick"
)

$ErrorActionPreference = "Stop"

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "MultiAgent V2 - Reliability Testing Suite" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan

# Set working directory to script location
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Check if OpenRouter API key is set
if (-not $env:OPENROUTER_API_KEY) {
    Write-Host "⚠️  Warning: OPENROUTER_API_KEY not set" -ForegroundColor Yellow
    $openRouterKey = Read-Host "   Please enter your OpenRouter API key (starts with 'sk-or-v1-')"
    if ($openRouterKey -match "^sk-or-v1-") {
        $env:OPENROUTER_API_KEY = $openRouterKey
        Write-Host "✅ OPENROUTER_API_KEY set successfully" -ForegroundColor Green
    } else {
        Write-Host "❌ Invalid OpenRouter API key format. Please ensure it starts with 'sk-or-v1-'" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "✅ OPENROUTER_API_KEY is already set." -ForegroundColor Green
}

# Ensure OPENAI_API_KEY is not set to force OpenRouter usage if it's not an OpenRouter key
if ($env:OPENAI_API_KEY -and -not ($env:OPENAI_API_KEY -match "^sk-or-v1-")) {
    Remove-Item Env:\OPENAI_API_KEY
    Write-Host "ℹ️  OPENAI_API_KEY unset to prioritize OpenRouter." -ForegroundColor Cyan
}

Write-Host "`nTest Type: $Type" -ForegroundColor Yellow
Write-Host "Logs will be saved to: ..\logs\reliability_testing\`n" -ForegroundColor Cyan

# Run the tests
python scripts\run_reliability_tests.py --type $Type -v

Write-Host "`n✅ Reliability tests complete!" -ForegroundColor Green
Write-Host "   Check logs in: ..\logs\reliability_testing\" -ForegroundColor Cyan

