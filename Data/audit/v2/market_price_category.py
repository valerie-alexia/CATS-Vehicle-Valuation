def market_category(stats):
    if stats.get("status") != "available":
        return None

    z = stats.get("posisi_prediksi_sd")
    sd = stats.get("sd")

    if z is None or sd is None or sd <= 0:
        return None

    if z < -3:
        return "Sangat Murah (Ekstrem)"
    if z < -2:
        return "Sangat Murah"
    if z < -1:
        return "Murah"
    if z <= 1:
        return "Wajar"
    if z <= 2:
        return "Mahal"
    if z <= 3:
        return "Sangat Mahal"

    return "Sangat Mahal (Ekstrem)"