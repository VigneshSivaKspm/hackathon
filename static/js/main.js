// Initialize tooltips
document.addEventListener("DOMContentLoaded", function () {
  const tooltipTriggerList = [].slice.call(
    document.querySelectorAll('[data-bs-toggle="tooltip"]')
  );
  tooltipTriggerList.map(function (tooltipTriggerEl) {
    return new bootstrap.Tooltip(tooltipTriggerEl);
  });

  // Initialize popovers
  const popoverTriggerList = [].slice.call(
    document.querySelectorAll('[data-bs-toggle="popover"]')
  );
  popoverTriggerList.map(function (popoverTriggerEl) {
    return new bootstrap.Popover(popoverTriggerEl);
  });

  // Handle emergency level indicators
  document.querySelectorAll(".emergency-level").forEach((el) => {
    const level = parseInt(el.dataset.level);
    el.classList.add(`bg-emergency-${level}`);

    if (level >= 4) {
      el.classList.add("critical-alert");
    }
  });

  // Real-time updates for dashboard (Global Alerts)
  if (document.getElementById("global-alerts-container")) {
    setupGlobalAlertsListener();
  }
});

// Function to listen for new global emergency alerts from Firebase
function setupGlobalAlertsListener() {
  const globalAlertsContainer = document.getElementById(
    "global-alerts-container"
  );
  const globalAlertsRef = firebase.database().ref("global_emergencies");

  globalAlertsRef.on("child_added", (snapshot) => {
    const alertData = snapshot.val();
    const alertDiv = document.createElement("div");
    alertDiv.className = "alert alert-info alert-dismissible fade show"; // Added dismissible
    alertDiv.innerHTML = `
      <strong>${
        alertData.type ? alertData.type.toUpperCase() : "EMERGENCY"
      }</strong>:
      ${alertData.location || "Unknown Location"}
      ${alertData.magnitude ? "(Magnitude: " + alertData.magnitude + ")" : ""}
      ${alertData.description ? "<br>" + alertData.description : ""}
      <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    globalAlertsContainer.prepend(alertDiv); // Show newest on top
  });
}

// Handle form submissions with fetch API
document.querySelectorAll("form[data-ajax]").forEach((form) => {
  form.addEventListener("submit", async function (e) {
    e.preventDefault();

    const submitBtn = form.querySelector('[type="submit"]');
    const originalText = submitBtn.innerHTML;
    submitBtn.disabled = true;
    submitBtn.innerHTML =
      '<span class="spinner-border spinner-border-sm" role="status"></span> Processing...';

    try {
      const formData = new FormData(form);
      const response = await fetch(form.action, {
        method: form.method,
        body: formData,
      });

      const result = await response.json();

      if (result.success) {
        // Show success message
        const alert = document.createElement("div");
        alert.className = "alert alert-success alert-dismissible fade show";
        alert.innerHTML = `
                          ${result.message || "Operation successful"}
                          <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                        `;
        form.prepend(alert);

        // Reset form if needed
        if (form.dataset.reset === "true") {
          form.reset();
        }
      } else {
        throw new Error(result.message || "Operation failed");
      }
    } catch (error) {
      const alert = document.createElement("div");
      alert.className = "alert alert-danger alert-dismissible fade show";
      alert.innerHTML = `
                          ${error.message}
                          <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                        `;
      form.prepend(alert);
    } finally {
      submitBtn.disabled = false;
      submitBtn.innerHTML = originalText;
    }
  });
});

// Handle location autocomplete
if (document.getElementById("location")) {
  const locationInput = document.getElementById("location");

  locationInput.addEventListener(
    "input",
    debounce(async function () {
      const query = this.value.trim();

      if (query.length < 3) return;

      try {
        const response = await fetch(
          `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(
            query
          )}`
        );
        const results = await response.json();

        // Show autocomplete dropdown
        showLocationSuggestions(results);
      } catch (error) {
        console.error("Location search error:", error);
      }
    }, 300)
  );
}

function showLocationSuggestions(results) {
  // Implement dropdown with location suggestions
  console.log("Location results:", results);
}

function debounce(func, wait) {
  let timeout;
  return function () {
    const context = this,
      args = arguments;
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(context, args), wait);
  };
}
