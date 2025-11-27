@echo off
echo ================================
echo   🔧 TESTING VALIDATION FIX
echo ================================

echo Testing AuraBot ask endpoint with corrected data...
powershell -Command ^
"$body = @{ ^
  session_id = 'validation_test_session'; ^
  question = 'How do I create a heading in HTML?'; ^
  html_context = $null; ^
  instructions_context = $null; ^
  user_id = $null ^
} | ConvertTo-Json; ^
try { ^
  $response = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/aurabot/ask' -Method POST -Body $body -ContentType 'application/json'; ^
  Write-Host '✅ SUCCESS! Validation Fixed!' -ForegroundColor Green; ^
  Write-Host 'Response received:' $response.response.Substring(0, 100) '...' -ForegroundColor Cyan; ^
  Write-Host 'Remaining attempts:' $response.remaining_attempts -ForegroundColor Yellow; ^
} catch { ^
  Write-Host '❌ Still failing:' $_.Exception.Message -ForegroundColor Red; ^
  Write-Host 'Response body:' $_.ErrorDetails.Message -ForegroundColor Yellow; ^
}"

pause
