document.getElementById('team1').addEventListener('change', updateStats);
document.getElementById('team2').addEventListener('change', updateStats);

function updateStats() {
    // 这里可以添加根据选择的队伍来更新统计数据的逻辑
    // 示例代码：
    let team1 = document.getElementById('team1').value;
    let team2 = document.getElementById('team2').value;

    if (team1 === '丹佛金塊') {
        document.getElementById('team1-stats').innerHTML = `
            <h2>40 胜</h2>
            <div>最大胜分: 129:98</div>
            <div>总得分: 10599</div>
            <div>场均得分: 105.99</div>
        `;
    }

    if (team2 === '波士顿塞尔提克') {
        document.getElementById('team2-stats').innerHTML = `
            <h2>60 胜</h2>
            <div>最大胜分: 114:76</div>
            <div>总得分: 10907</div>
            <div>场均得分: 109.07</div>
        `;
    }
}
