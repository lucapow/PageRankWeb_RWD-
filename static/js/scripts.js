function updateChart(imageId, route, rangeSelectId, rankSelectId) {
    const range = document.getElementById(rangeSelectId).value;
    const rank = document.getElementById(rankSelectId).value;
    const imageUrl = `/${route}?range=${range}&rank=${rank}&_=${new Date().getTime()}`;
    document.getElementById(imageId).src = imageUrl;
}

function updateScoreTrendChart() {
    updateChart('chart1', 'generate_score_trend_chart', 'rangeSelect', 'rankSelect');
}

function updateOfficialRankTrendChart() {
    updateChart('chart7', 'generate_official_rank_trend_chart', 'rangeSelect', 'rankSelect');
}