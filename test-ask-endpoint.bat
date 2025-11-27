@echo off
echo ================================
echo   TESTING AURABOT ASK ENDPOINT
echo ================================

timeout /t 3 /nobreak

echo Testing ask question with sample data...
powershell -Command ^
"$body = @{ ^
  session_id = 'test_session_456'; ^
  question = 'How do I create a heading in HTML?'; ^
  html_context = '<h1>Test</h1>'; ^
  instructions_context = 'Create a heading element' ^
} | ConvertTo-Json; ^
try { ^
  $response = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/aurabot/ask' -Method POST -Body $body -ContentType 'application/json'; ^
  Write-Host 'SUCCESS: AuraBot Ask Endpoint Working!'; ^
  Write-Host 'Question asked successfully!'; ^
  Write-Host 'Remaining attempts:' $response.remaining_attempts; ^
  Write-Host 'Response preview:' $response.response.Substring(0, 100) ^
} catch { ^
  Write-Host 'Ask Endpoint Result:' $_.Exception.Message ^
}"

echo.
echo ================================
echo Test complete!
pause
