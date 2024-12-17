// chart.js 文件中的代碼
document.addEventListener('DOMContentLoaded', function() {
    var ctxHome = document.getElementById('homeTeamChart').getContext('2d');
    var ctxAway = document.getElementById('awayTeamChart').getContext('2d');

    // 主場隊伍圖表
    new Chart(ctxHome, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [33, 67],  // 示例數據，應由後端提供
                backgroundColor: ['#ff6384', '#cccccc']
            }],
            labels: ['勝率', '其他']
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });

    // 客場隊伍圖表
    new Chart(ctxAway, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [67, 33],  // 示例數據，應由後端提供
                backgroundColor: ['#36a2eb', '#cccccc']
            }],
            labels: ['勝率', '其他']
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
});
