import os
from datetime import datetime
import json

class TrainingLogger:
    def __init__(self, history_file='training_history.txt'):
        self.history_file = history_file
        self.current_session = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'epochs': []
        }
    
    def log_epoch(self, epoch_num, metrics):
        """Log metrics for a single epoch"""
        epoch_data = {
            'epoch': epoch_num,
            'train_loss': metrics['train_loss'],
            'train_disaster_acc': metrics['train_disaster_acc'],
            'train_severity_acc': metrics['train_severity_acc'],
            'val_loss': metrics['val_loss'],
            'val_disaster_acc': metrics['val_disaster_acc'],
            'val_severity_acc': metrics['val_severity_acc']
        }
        self.current_session['epochs'].append(epoch_data)
    
    def log_final_metrics(self, test_metrics):
        """Log final test metrics"""
        self.current_session['test_metrics'] = {
            'test_disaster_acc': test_metrics['disaster_acc'],
            'test_severity_acc': test_metrics['severity_acc']
        }
    
    def save_session(self):
        """Save current training session to file"""
        header = f"\n{'='*80}\n"
        header += f"Training Session: {self.current_session['timestamp']}\n"
        header += f"{'='*80}\n"
        
        epoch_logs = []
        for epoch in self.current_session['epochs']:
            epoch_log = (
                f"Epoch {epoch['epoch']+1}:\n"
                f"  Train - Loss: {epoch['train_loss']:.4f}, "
                f"Disaster Acc: {epoch['train_disaster_acc']:.4f}, "
                f"Severity Acc: {epoch['train_severity_acc']:.4f}\n"
                f"  Val   - Loss: {epoch['val_loss']:.4f}, "
                f"Disaster Acc: {epoch['val_disaster_acc']:.4f}, "
                f"Severity Acc: {epoch['val_severity_acc']:.4f}\n"
            )
            epoch_logs.append(epoch_log)
        
        final_metrics = (
            f"\nFinal Test Results:\n"
            f"  Disaster Accuracy: {self.current_session['test_metrics']['test_disaster_acc']:.4f}\n"
            f"  Severity Accuracy: {self.current_session['test_metrics']['test_severity_acc']:.4f}\n"
        )
        
        footer = f"{'='*80}\n"
        
        with open(self.history_file, 'a') as f:
            f.write(header)
            f.write('\n'.join(epoch_logs))
            f.write(final_metrics)
            f.write(footer)
        
        # Also save as JSON for programmatic access
        json_file = self.history_file.replace('.txt', '.json')
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        history.append(self.current_session)
        with open(json_file, 'w') as f:
            json.dump(history, f, indent=2)