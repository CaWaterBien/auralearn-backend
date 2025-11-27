<?php

namespace Database\Seeders;

use App\Models\User;
use Illuminate\Database\Seeder;
use Illuminate\Support\Facades\Hash;

class UserSeeder extends Seeder
{
    public function run(): void
    {
        $users = [
            [
                'name' => 'John Doe',
                'full_name' => 'John Doe',
                'email' => 'john@example.com',
                'password' => Hash::make('password123'),
                'join_date' => '2024-01-15',
                'avatar' => 'https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=150&h=150&fit=crop&crop=face',
                'progress' => [
                    'completedCourses' => 8,
                    'totalCourses' => 15,
                    'currentCourse' => 'Advanced CSS Grid',
                    'streak' => 12,
                    'totalPoints' => 2850,
                    'rank' => 15,
                    'badges' => ['HTML Master', 'CSS Expert', '30-Day Streak', 'Quick Learner'],
                    'recentLessons' => [
                        ['title' => 'CSS Grid Layout', 'progress' => 75, 'lastAccessed' => '2024-01-20'],
                        ['title' => 'Flexbox Fundamentals', 'progress' => 100, 'lastAccessed' => '2024-01-19'],
                        ['title' => 'Responsive Design', 'progress' => 50, 'lastAccessed' => '2024-01-18']
                    ]
                ],
                'preferences' => [
                    'theme' => 'light',
                    'notifications' => true,
                    'language' => 'en'
                ],
                'is_active' => 1,
            ],
            [
                'name' => 'Jane Smith',
                'full_name' => 'Jane Smith',
                'email' => 'jane@example.com',
                'password' => Hash::make('password123'),
                'join_date' => '2024-02-01',
                'avatar' => 'https://images.unsplash.com/photo-1494790108755-2616b6967e2b?w=150&h=150&fit=crop&crop=face',
                'progress' => [
                    'completedCourses' => 12,
                    'totalCourses' => 15,
                    'currentCourse' => 'JavaScript Advanced',
                    'streak' => 25,
                    'totalPoints' => 4200,
                    'rank' => 8,
                    'badges' => ['JavaScript Ninja', 'HTML Master', 'CSS Expert', 'Project Builder', '60-Day Streak'],
                    'recentLessons' => [
                        ['title' => 'Async JavaScript', 'progress' => 60, 'lastAccessed' => '2024-01-20'],
                        ['title' => 'DOM Manipulation', 'progress' => 100, 'lastAccessed' => '2024-01-19'],
                        ['title' => 'ES6 Features', 'progress' => 90, 'lastAccessed' => '2024-01-18']
                    ]
                ],
                'preferences' => [
                    'theme' => 'dark',
                    'notifications' => true,
                    'language' => 'en'
                ],
                'is_active' => 1,
            ],
            [
                'name' => 'Demo User',
                'full_name' => 'Demo User',
                'email' => 'demo@auralearn.com',
                'password' => Hash::make('password123'),
                'join_date' => '2024-01-01',
                'avatar' => 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150&h=150&fit=crop&crop=face',
                'progress' => [
                    'completedCourses' => 5,
                    'totalCourses' => 15,
                    'currentCourse' => 'HTML Fundamentals',
                    'streak' => 7,
                    'totalPoints' => 1200,
                    'rank' => 42,
                    'badges' => ['First Steps', 'HTML Beginner', 'Weekly Learner'],
                    'recentLessons' => [
                        ['title' => 'HTML Forms', 'progress' => 30, 'lastAccessed' => '2024-01-20'],
                        ['title' => 'HTML Elements', 'progress' => 100, 'lastAccessed' => '2024-01-19'],
                        ['title' => 'HTML Structure', 'progress' => 100, 'lastAccessed' => '2024-01-17']
                    ]
                ],
                'preferences' => [
                    'theme' => 'light',
                    'notifications' => false,
                    'language' => 'en'
                ],
                'is_active' => 1,
            ]
        ];

        foreach ($users as $userData) {
            $user = User::updateOrCreate(
                ['email' => $userData['email']],
                [
                    'name' => $userData['name'],
                    'full_name' => $userData['full_name'],
                    'password' => $userData['password'],
                    'join_date' => $userData['join_date'],
                    'avatar' => $userData['avatar'],
                    'is_active' => true,
                ]
            );
            
            // Update JSON fields separately to avoid casting issues
            $user->update([
                'progress' => $userData['progress'],
                'preferences' => $userData['preferences']
            ]);
        }
    }
}