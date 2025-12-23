import React from 'react';
import { Activity } from 'lucide-react';

export const Header = () => {
    return (
        <header className="w-full p-4 flex items-center justify-between bg-white/80 backdrop-blur-md border-b border-gray-200 sticky top-0 z-50">
            <div className="flex items-center gap-2">
                <div className="w-8 h-8 bg-ios-blue rounded-lg flex items-center justify-center text-white">
                    <Activity size={20} />
                </div>
                <h1 className="text-xl font-semibold text-gray-900 tracking-tight">ROS2 Control</h1>
            </div>
            <div className="text-sm text-gray-500 font-medium">
                v1.0
            </div>
        </header>
    );
};
