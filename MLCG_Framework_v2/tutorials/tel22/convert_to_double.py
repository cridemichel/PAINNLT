import torch

# Load the float32 model
model = torch.jit.load('tel22_model.pt', map_location='cpu')

# Convert the entire model (parameters, buffers) to float64
model = model.to(torch.float64)

# Save the float64 model
torch.jit.save(model, 'tel22_model_float64.pt')
print("Model converted to float64 and saved as tel22_model_float64.pt")
