using FileIO, ImageIO, Images

root_dir = dirname(Base.current_project())

epochs = load(joinpath(root_dir, "plots/training/epoch1.png"))
base_shape = (size(epochs,1),size(epochs,2),1)
let epochs = reshape(epochs, base_shape)
for i in 2:42
    epochs = cat(epochs, reshape(load(joinpath(root_dir, "plots/training/epoch$i.png")),base_shape), dims=3)
end

# Save animation as GIF
save(joinpath(root_dir, "plots/training/UDE_training.gif"), epochs, fps=5)
end # let