<?php

namespace Database\Seeders;

use Illuminate\Database\Seeder;
use Illuminate\Support\Facades\Hash;
use App\Models\Admin;

class AdminSeeder extends Seeder
{
    public function run(): void
    {
        Admin::updateOrCreate(
            ['email' => 'admin@auralearn.local'],
            [
                'name' => 'AuraLearn Admin',
                'password' => Hash::make('Admin123!'),
            ]
        );
    }
} 