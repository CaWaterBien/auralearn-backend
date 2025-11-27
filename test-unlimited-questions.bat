@echo off
echo Testing AuraBot with Unlimited Questions...
echo.

echo 1. Testing Activity 1 Session:
powershell -Command "$body = '{\"session_id\":\"activity_1_test\",\"question\":\"How do I create a heading in HTML?\",\"html_context\":null,\"instructions_context\":null,\"user_id\":null}'; $headers = @{\"Content-Type\"=\"application/json\"}; try { $response = Invoke-RestMethod -Uri \"http://127.0.0.1:8000/api/aurabot/ask\" -Method POST -Headers $headers -Body $body; Write-Host \"SUCCESS: Activity 1 AuraBot Response:\" -ForegroundColor Green; Write-Host $response.response.Substring(0,150) \"...\" -ForegroundColor Cyan; Write-Host \"Remaining attempts: $($response.remaining_attempts)\" -ForegroundColor Yellow } catch { Write-Host \"Error: $($_.Exception.Message)\" -ForegroundColor Red }"

echo.
echo 2. Testing Activity 2 Session (Different History):
powershell -Command "$body = '{\"session_id\":\"activity_2_test\",\"question\":\"What is a paragraph tag?\",\"html_context\":null,\"instructions_context\":null,\"user_id\":null}'; $headers = @{\"Content-Type\"=\"application/json\"}; try { $response = Invoke-RestMethod -Uri \"http://127.0.0.1:8000/api/aurabot/ask\" -Method POST -Headers $headers -Body $body; Write-Host \"SUCCESS: Activity 2 AuraBot Response:\" -ForegroundColor Green; Write-Host $response.response.Substring(0,150) \"...\" -ForegroundColor Cyan; Write-Host \"Remaining attempts: $($response.remaining_attempts)\" -ForegroundColor Yellow } catch { Write-Host \"Error: $($_.Exception.Message)\" -ForegroundColor Red }"

echo.
echo 3. Testing Multiple Questions (Should be unlimited):
powershell -Command "$body = '{\"session_id\":\"activity_1_test\",\"question\":\"What about multiple headings?\",\"html_context\":null,\"instructions_context\":null,\"user_id\":null}'; $headers = @{\"Content-Type\"=\"application/json\"}; try { $response = Invoke-RestMethod -Uri \"http://127.0.0.1:8000/api/aurabot/ask\" -Method POST -Headers $headers -Body $body; Write-Host \"SUCCESS: Multiple Questions Work!\" -ForegroundColor Green; Write-Host \"Remaining attempts: $($response.remaining_attempts)\" -ForegroundColor Yellow } catch { Write-Host \"Error: $($_.Exception.Message)\" -ForegroundColor Red }"

echo.
echo ===== ALL FIXES TESTED =====
