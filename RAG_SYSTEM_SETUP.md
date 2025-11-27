# 🤖 AuraBot RAG System - Quick Start Guide

## 🚨 IMPORTANT: Required API Keys

Before starting, you MUST obtain these API keys:

1. **OpenAI API Key** (Required for embeddings)
   - Go to: https://platform.openai.com/api-keys
   - Create a new API key
   - Add to `.env` as: `OPENAI_API_KEY=your_key_here`

2. **Nebius API Key** ✅ Already configured!
   - Pre-configured in the system
   - Uses OpenAI GPT-OSS-20B model

## 🚀 Quick Setup (5 Minutes)

### Option 1: Automated Setup
```bash
# Windows
./setup-rag.bat

# Linux/Mac  
./setup-rag.sh
```

### Option 2: Manual Setup
```bash
# 1. Install packages
composer install

# 2. Setup environment
cp .env.example .env
# IMPORTANT: Add your OPENAI_API_KEY to .env

# 3. Run migrations
php artisan migrate

# 4. Create sample data
php artisan rag:create-samples

# 5. Test system
php artisan rag:test
```

## ✅ Verification Steps

### 1. Test Database Connection
```bash
php artisan tinker
# In tinker: \App\Models\RagDocument::count()
# Should return: number of documents > 0
```

### 2. Test Nebius API
```bash
php artisan rag:test --question="What is HTML?"
# Should show: ✅ All tests passed!
```

### 3. Test Frontend Integration
```bash
# Start backend
php artisan serve

# Start frontend (new terminal)
cd ../capstone-app
npm run dev

# Visit: http://localhost:3000
# Click AuraBot icon and ask a question
```

## 🎯 How It Works

### User Experience
1. **User clicks AuraBot** → Floating chatbot opens
2. **User asks question** → Context automatically extracted from page
3. **RAG system activates** → Searches knowledge base for relevant content
4. **AI generates hint** → Nebius API creates educational response
5. **Response delivered** → Hint provided, attempt counter updated
6. **Conversation saved** → Full history maintained

### Backend Flow
```
Question → Context Extraction → RAG Search → AI Generation → Response
    ↓
Session Management → Attempt Tracking → Conversation Storage
```

## 📊 Expected Results

After setup, you should see:
- **AuraBot icon** in bottom-right corner of frontend
- **Attempt counter** showing "3" initially  
- **Intelligent responses** based on course content
- **Context awareness** of current HTML/CSS work
- **Educational hints** instead of direct answers

## 🔧 Configuration Options

### Customize in `.env`:
```env
# Attempt limits
AURABOT_ATTEMPT_LIMIT=3        # Questions per session
AURABOT_MAX_TOKENS=5000        # Max response length

# RAG settings
RAG_MAX_CHUNKS=5               # Documents per search
RAG_CHUNK_SIZE=1000            # Text chunk size
```

## 🆘 Troubleshooting

### "No response from AuraBot"
1. Check Laravel server is running: `php artisan serve`
2. Verify API keys in `.env`
3. Test with: `php artisan rag:test`

### "Question limit reached immediately"
1. Check session storage in database
2. Reset session: Admin panel → Reset Session
3. Or wait 1 hour for automatic reset

### "Embedding generation failed"
1. Verify `OPENAI_API_KEY` in `.env`
2. Check OpenAI API quota/billing
3. Test: `php artisan rag:create-samples`

## 🎉 Success Confirmation

Your system is working correctly if:
- ✅ `php artisan rag:test` passes all tests
- ✅ AuraBot responds with educational hints  
- ✅ Attempt counter decreases after questions
- ✅ Conversation history persists
- ✅ HTML context is detected automatically

## 📞 Support

If you encounter any issues:
1. Check the comprehensive documentation: `AURABOT_RAG_SYSTEM_DOCUMENTATION.md`
2. Run system diagnostics: `php artisan rag:test`
3. Check logs: `storage/logs/laravel.log`

---

🚀 **Your AuraBot RAG system is now ready to provide intelligent, educational assistance to your learners!**

