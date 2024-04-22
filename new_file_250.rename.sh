counter = 0
for file in *; do
    new_name="new_file_$counter.$file"
    mv "$file" "$new_name"
    counter=$((counter + 1))
done