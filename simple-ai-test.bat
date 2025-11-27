@echo off
echo ====================================
echo Simple AI Validation Test
echo ====================================
echo.

echo Testing AI validation with simple PowerShell approach...
echo.

powershell -Command "try { $response = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/activities/2/submit' -Method POST -ContentType 'application/json' -Body '{\"user_code\":\"Hello World\",\"time_spent_minutes\":1}'; Write-Host 'SUCCESS! Response received' -ForegroundColor Green; Write-Host 'Score:' $response.score; Write-Host 'Status:' $response.completion_status; Write-Host 'AI Powered:' $response.ai_powered } catch { Write-Host 'Error:' $_.Exception.Message -ForegroundColor Red }"

echo.
echo ====================================
echo Test Complete!
echo ====================================
pause
