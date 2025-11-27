@echo off
echo ====================================
echo Testing AI-Powered Submission Validation
echo ====================================
echo.
echo This test verifies that AI validation is working correctly
echo for activity submissions with intelligent feedback.
echo.

echo Step 1: Testing AI validation with COMPLETE HTML code
powershell -Command "$body = @{user_code='^^^<!DOCTYPE html^^^>^^^<html^^^>^^^<head^^^>^^^<title^^^>My First Image^^^</title^^^>^^^</head^^^>^^^<body^^^>^^^<img src=^^^\"https://via.placeholder.com/300^^^\" alt=^^^\"Sample Image^^^\"^^^>^^^</body^^^>^^^</html^^^>'; time_spent_minutes=10} | ConvertTo-Json; try { $headers = @{'Content-Type'='application/json'}; $response = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/activities/2/submit' -Method POST -Headers $headers -Body $body; Write-Host '✅ AI Validation SUCCESS!' -ForegroundColor Green; Write-Host 'Score:' $response.score -ForegroundColor Cyan; Write-Host 'AI Powered:' $response.ai_powered -ForegroundColor Yellow; Write-Host 'Status:' $response.completion_status -ForegroundColor Magenta; Write-Host 'Message:' $response.message -ForegroundColor White } catch { Write-Host '❌ Error:' $_.Exception.Message -ForegroundColor Red }"

echo.
echo Step 2: Testing AI validation with INCOMPLETE HTML code  
powershell -Command "$body = @{user_code='^^^<html^^^>^^^<head^^^>^^^</head^^^>^^^<body^^^>Hello World^^^</body^^^>^^^</html^^^>'; time_spent_minutes=5} | ConvertTo-Json; try { $headers = @{'Content-Type'='application/json'}; $response = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/activities/2/submit' -Method POST -Headers $headers -Body $body; Write-Host '✅ AI Validation SUCCESS!' -ForegroundColor Green; Write-Host 'Score:' $response.score -ForegroundColor Cyan; Write-Host 'AI Powered:' $response.ai_powered -ForegroundColor Yellow; Write-Host 'Status:' $response.completion_status -ForegroundColor Magenta; Write-Host 'Message:' $response.message -ForegroundColor White } catch { Write-Host '❌ Error:' $_.Exception.Message -ForegroundColor Red }"

echo.
echo Step 3: Testing AI validation with MINIMAL HTML code
powershell -Command "$body = @{user_code='^^^<h1^^^>Hello^^^</h1^^^>'; time_spent_minutes=2} | ConvertTo-Json; try { $headers = @{'Content-Type'='application/json'}; $response = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/activities/2/submit' -Method POST -Headers $headers -Body $body; Write-Host '✅ AI Validation SUCCESS!' -ForegroundColor Green; Write-Host 'Score:' $response.score -ForegroundColor Cyan; Write-Host 'AI Powered:' $response.ai_powered -ForegroundColor Yellow; Write-Host 'Status:' $response.completion_status -ForegroundColor Magenta; Write-Host 'Message:' $response.message -ForegroundColor White } catch { Write-Host '❌ Error:' $_.Exception.Message -ForegroundColor Red }"

echo.
echo Step 4: Checking submission status after AI validation
powershell -Command "try { $response = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/activities/2/status' -Method GET; Write-Host '✅ Status Check SUCCESS!' -ForegroundColor Green; Write-Host 'Total Attempts:' $response.total_attempts -ForegroundColor Cyan; Write-Host 'Is Completed:' $response.is_completed -ForegroundColor Yellow; Write-Host 'Best Score:' $response.best_score -ForegroundColor Magenta } catch { Write-Host '❌ Error:' $_.Exception.Message -ForegroundColor Red }"

echo.
echo ====================================
echo AI Validation Testing Complete!
echo ====================================
echo.
echo Expected Results:
echo - Step 1: High score (80%+), 'passed' status, AI-powered feedback
echo - Step 2: Medium score (40-70%), 'partial' status, specific suggestions
echo - Step 3: Low score (0-40%), 'failed' status, detailed improvements
echo - Step 4: Updated attempt count and best score tracking
echo.
pause
