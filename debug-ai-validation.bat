@echo off
echo Debugging AI Validation System
echo ================================
echo.

echo Step 1: Clear Laravel logs
powershell -Command "Clear-Content storage\logs\laravel.log -Force -ErrorAction SilentlyContinue"
echo Logs cleared.

echo.
echo Step 2: Test simple submission
curl -X POST "http://127.0.0.1:8000/api/activities/2/submit" -H "Content-Type: application/json" -d "{\"user_code\":\"Hello World\",\"time_spent_minutes\":1}"

echo.
echo Step 3: Check Laravel logs for errors
powershell -Command "if (Test-Path storage\logs\laravel.log) { Get-Content storage\logs\laravel.log | Select-Object -Last 10 } else { Write-Host 'No log file found' }"

echo.
pause
