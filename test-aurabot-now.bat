@echo off
echo ================================
echo   TESTING AURABOT RAG SYSTEM
echo ================================
echo.

echo Waiting for server to start...
timeout /t 5 /nobreak > nul

echo Testing health endpoint...
powershell -Command "try { $response = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/aurabot/health' -Method GET; Write-Host 'SUCCESS: Health endpoint working!'; Write-Host 'Status:' $response.status; Write-Host 'Database:' $response.database } catch { Write-Host 'Health ERROR:' $_.Exception.Message }"

echo.
echo Testing session status...
powershell -Command "try { $response = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/aurabot/session-status?session_id=test123' -Method GET; Write-Host 'SUCCESS: Session status working!'; Write-Host 'Can ask:' $response.can_ask; Write-Host 'Attempts:' $response.remaining_attempts } catch { Write-Host 'Session ERROR:' $_.Exception.Message }"

echo.
echo Testing ask question endpoint (without OpenAI key - will show meaningful error)...
powershell -Command "try { $body = @{ session_id='test123'; question='What is HTML?' } | ConvertTo-Json; $response = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/aurabot/ask' -Method POST -Body $body -ContentType 'application/json'; Write-Host 'SUCCESS: Ask endpoint reached!'; Write-Host 'Response:' $response.response } catch { Write-Host 'Ask result:' $_.Exception.Message }"

echo.
echo ================================
echo TEST COMPLETE!
echo.
echo If you see "SUCCESS" messages above, the system is working!
echo Add OPENAI_API_KEY to .env to enable full functionality.
echo ================================
pause
