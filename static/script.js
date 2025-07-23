document.addEventListener("DOMContentLoaded", function () {
  const formulario = document.getElementById("formulario");
  const inputs = formulario.querySelectorAll("input, select");
  const boton = document.getElementById("btnPredecir");
  const resultado = document.getElementById("resultado");
  const bloqueResultado = document.getElementById("bloque-resultado");
  const btnOtraPrediccion = document.getElementById("btnOtraPrediccion");

  function validarCampos() {
    let validos = true;
    inputs.forEach((input) => {
      if (!input.checkValidity() || input.value.trim() === "") {
        validos = false;
      }
    });
    boton.disabled = !validos;
  }

  inputs.forEach((input) => {
    input.addEventListener("input", () => {
      if (input.checkValidity() && input.value.trim() !== "") {
        input.classList.add("valid");
      } else {
        input.classList.remove("valid");
      }
      validarCampos();
    });
  });

  formulario.addEventListener("submit", async function (e) {
    e.preventDefault();

    const formData = new FormData(formulario);
    const valores = [];

    for (let pair of formData.entries()) {
      valores.push(pair[1]);
    }

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ valores }),
      });

      if (!response.ok) {
        throw new Error("Error en la predicción");
      }

      const data = await response.json();

      formulario.style.display = "none";

      document.getElementById("titulo").innerText =
        "RESULTADOS DE LA PREDICCIÓN";
      document.getElementById("subtitulo").innerText =
        "Según los datos ingresados, este es el resultado estimado:";

      bloqueResultado.style.display = "block";

      resultado.innerHTML = `
        <div>
          <strong style="color: red;">Costos estimados:</strong><br>
          <span style="color: black; font-size: 1.2em;">$${data.Costos_MillonesCOP} millones COP</span>
        </div>
      `;
    } catch (error) {
      resultado.innerText = "Error al realizar la predicción.";
      console.error("Error:", error);
    }
  });

  btnOtraPrediccion.addEventListener("click", () => {
    window.location.reload();
  });

  validarCampos();
});