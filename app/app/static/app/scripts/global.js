function getCookie(name) {
    const cookieValue = document.cookie
        .split(";")
        .map((cookie) => cookie.trim())
        .find((cookie) => cookie.startsWith(name + "="));

    if (cookieValue) {
        return decodeURIComponent(cookieValue.split("=")[1]);
    } else {
        return null;
    }
}

function openModal(name) {
    var id_name = "#" + name
    $(id_name).modal("toggle");
}

function closeModal(name) {
    var id_name = "#" + name
    $(id_name).modal("hide");
}
