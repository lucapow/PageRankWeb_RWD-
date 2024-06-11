document.getElementById('team1').addEventListener('change', updateStats);
document.getElementById('team2').addEventListener('change', updateStats);

function updateStats() {
    // 这里可以添加根据选择的队伍来更新统计数据的逻辑
    // 示例代码：
    let team1 = document.getElementById('team1').value;
    let team2 = document.getElementById('team2').value;
    if (team1 === '--選擇學校--') {
        document.getElementById('team1-stats').innerHTML = `
            <h2>勝利數</h2>
            <div>最大勝分: --</div>
            <div>總得分: --</div>
            <div>場均得分: --</div>
        `;
    }
    if (team1 === '平鎮高中') {
        document.getElementById('team1-stats').innerHTML = `
            <h2>40 勝</h2>
            <div>最大勝分: 129:98</div>
            <div>總得分: 10599</div>
            <div>場均得分: 105.99</div>
        `;
    }
    if (team1 === '穀保家商') {
        document.getElementById('team1-stats').innerHTML = `
            <h2>40 勝</h2>
            <div>最大勝分: 129:98</div>
            <div>總得分: 10599</div>
            <div>場均得分: 105.99</div>
        `;
    }
    if (team1 === '高苑工商') {
        document.getElementById('team1-stats').innerHTML = `
            <h2>40 勝</h2>
            <div>最大勝分: 129:98</div>
            <div>總得分: 10599</div>
            <div>場均得分: 105.99</div>
        `;
    }
    if (team1 === '鶯歌工商') {
        document.getElementById('team1-stats').innerHTML = `
            <h2>40 勝</h2>
            <div>最大勝分: 129:98</div>
            <div>總得分: 10599</div>
            <div>場均得分: 105.99</div>
        `;
    }
    if (team2 === '--選擇學校--') {
        document.getElementById('team2-stats').innerHTML = `
            <h2>勝利數</h2>
            <div>最大勝分: --</div>
            <div>總得分: --</div>
            <div>場均得分: --</div>
        `;
    }
    if (team2 === '平鎮高中') {
        document.getElementById('team2-stats').innerHTML = `
            <h2>60 勝</h2>
            <div>最大勝分: 129:98</div>
            <div>總得分: 10599</div>
            <div>場均得分: 105.99</div>
        `;
    }
    if (team2 === '穀保家商') {
        document.getElementById('team2-stats').innerHTML = `
             <h2>60 勝</h2>
            <div>最大勝分: 129:98</div>
            <div>總得分: 10599</div>
            <div>場均得分: 105.99</div>
        `;
    }
    if (team2 === '高苑工商') {
        document.getElementById('team2-stats').innerHTML = `
             <h2>60 勝</h2>
            <div>最大勝分: 129:98</div>
            <div>總得分: 10599</div>
            <div>場均得分: 105.99</div>
        `;
    }
    if (team2 === '鶯歌工商') {
        document.getElementById('team2-stats').innerHTML = `
             <h2>60 勝</h2>
            <div>最大勝分: 129:98</div>
            <div>總得分: 10599</div>
            <div>場均得分: 105.99</div>
        `;
    }
}
