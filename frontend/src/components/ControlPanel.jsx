import React from 'react';
import { motion } from 'framer-motion';
import { Power, Play, Square } from 'lucide-react';
import { clsx } from 'clsx';

export const ControlPanel = ({ isConnected, isRunning, onConnect, onRun, onStop }) => {
    return (
        <div className="w-full max-w-md mx-auto p-6 bg-white rounded-3xl shadow-xl border border-gray-100">
            <div className="grid grid-cols-3 gap-4">
                <ControlButton
                    onClick={onConnect}
                    disabled={isConnected}
                    icon={<Power size={24} />}
                    label={isConnected ? "Connected" : "Connect"}
                    color="blue"
                    isActive={isConnected}
                />
                <ControlButton
                    onClick={onRun}
                    disabled={!isConnected || isRunning}
                    icon={<Play size={24} />}
                    label="Run"
                    color="green"
                    isActive={isRunning}
                />
                <ControlButton
                    onClick={onStop}
                    disabled={!isConnected || !isRunning}
                    icon={<Square size={24} />}
                    label="Stop"
                    color="red"
                    isActive={false}
                />
            </div>
        </div>
    );
};

const ControlButton = ({ onClick, disabled, icon, label, color, isActive }) => {
    const colors = {
        blue: 'bg-ios-blue text-white shadow-ios-blue/30',
        green: 'bg-ios-green text-white shadow-ios-green/30',
        red: 'bg-ios-red text-white shadow-ios-red/30',
        gray: 'bg-gray-100 text-gray-400',
    };

    const activeClass = disabled ? colors.gray : colors[color];

    return (
        <motion.button
            whileTap={{ scale: 0.95 }}
            onClick={onClick}
            disabled={disabled}
            className={clsx(
                "flex flex-col items-center justify-center p-4 rounded-2xl transition-all duration-300 shadow-lg",
                activeClass,
                disabled ? "cursor-not-allowed opacity-50 shadow-none" : "hover:brightness-110"
            )}
        >
            <div className="mb-2">{icon}</div>
            <span className="text-sm font-medium">{label}</span>
        </motion.button>
    );
};
