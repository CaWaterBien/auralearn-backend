<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::table('activity_submissions', function (Blueprint $table) {
            $table->text('code_explanation')->nullable()->after('error_details');
            $table->integer('explanation_word_count')->default(0)->after('code_explanation');
            $table->boolean('explanation_required')->default(true)->after('explanation_word_count');
            $table->json('explanation_analysis')->nullable()->after('explanation_required'); // AI analysis of explanation quality
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('activity_submissions', function (Blueprint $table) {
            $table->dropColumn(['code_explanation', 'explanation_word_count', 'explanation_required', 'explanation_analysis']);
        });
    }
};
