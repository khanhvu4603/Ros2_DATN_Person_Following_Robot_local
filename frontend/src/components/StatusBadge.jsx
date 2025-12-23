import React from 'react';
import { clsx } from 'clsx';

export const StatusBadge = ({ status }) => {
    const getStatusColor = (s) => {
        switch (s) {
            case 'connected':
            case 'running':
                return 'bg-ios-green text-white';
            case 'connecting':
                return 'bg-yellow-500 text-white';
            case 'disconnected':
            case 'stopped':
            default:
                return 'bg-gray-400 text-white';
        }
    };

    return (
        <div className={clsx(
            "px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wide shadow-sm transition-all duration-300",
            getStatusColor(status)
        )}>
            {status}
        </div>
    );
};
