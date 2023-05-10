const fileInput = document.getElementById("file");
const fileLabel = document.getElementById("file-label");

fileInput.addEventListener("change", function() {
  if (this.files && this.files[0]) {
    fileLabel.textContent = this.files[0].name;
  }
});
