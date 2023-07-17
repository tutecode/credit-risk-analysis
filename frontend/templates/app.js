let previousOption = null;

function checkOptions() {
  const optionA = document.getElementById("optionA");
  const optionB = document.getElementById("optionB");

  if (previousOption === optionA && optionB.checked) {
    optionA.checked = false;
  }

  previousOption = document.querySelector('input[name="option"]:checked');
}

const optionA = document.getElementById("optionA");
const optionB = document.getElementById("optionB");

optionA.addEventListener("click", checkOptions);
optionB.addEventListener("click", checkOptions);