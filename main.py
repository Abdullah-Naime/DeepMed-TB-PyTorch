import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import database

def load_trained_model():
    """Load the fine-tuned ResNet-18 model weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("tb_vision_resnet18.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, device

def main():
    print("==========================================")
    print("          TBVision - CLI Interface        ")
    print("==========================================")
    
    database.init_db()
    
    while True:
        print("\nOptions:")
        print("1. View Database Scan Logs")
        print("2. Simulate Test Sample Inference & Log")
        print("3. Exit")
        
        choice = input("Select an option (1-3): ").strip()
        
        if choice == "1":
            logs = database.get_all_logs()
            print(f"\n--- Database Scan History ({len(logs)} records) ---")
            if not logs:
                print("No records found yet.")
            for log in logs:
                print(f"ID: {log[0]} | File: {log[1]} | True: {log[2]} | Pred: {log[3]} | Conf: {log[4]:.2f} | Status: {log[5]} | Time: {log[6]}")
                
        elif choice == "2":
            print("\nSimulating prediction on a sample test case...")
            # Example logging a test interaction
            database.log_prediction(
                filename="MCUCXR_Sample_Scan.png",
                true_label="TB-related",
                predicted_label="TB-related",
                confidence=0.9123,
                status="Correct"
            )
            print("Successfully ran simulation and logged result to SQLite database!")
            
        elif choice == "3":
            print("Exiting TBVision. Stay healthy!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
