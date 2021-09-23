from model import VAE

model_load_directory = "saved_models/first"
vae = VAE.load(model_load_directory)
print("loaded model")