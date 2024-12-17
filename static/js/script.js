// document.addEventListener("DOMContentLoaded", () => {
//     const yearSelect = document.getElementById('yearSelect');
//     const tableBody = document.getElementById("teamsTableBody");

//     // 檢查必要的 DOM 元素是否存在
//     if (!tableBody) {
//         console.error("Error: 'teamsTableBody' element not found in the DOM.");
//         return;
//     }

//     if (yearSelect) {
//         yearSelect.addEventListener('change', filterData);
//     } else {
//         console.error("Error: 'yearSelect' element not found in the DOM.");
//         return;
//     }
// });

// function filterData() {
//     const selectedYear = document.getElementById("yearSelect").value;

//     if (!selectedYear || selectedYear === '--選擇年份--') {
//         alert('請選擇有效的年份');
//         return;
//     }

//     // 發送請求獲取指定年份的數據
//     fetch(`/api/calculate_pr_rank?year=${selectedYear}`)
//         .then(response => {
//             console.log("Response Status:", response.status);
//             if (!response.ok) {
//                 return response.json().then(errData => {
//                     throw new Error(errData.error || errData.message || "未知錯誤");
//                 });
//             }
//             return response.json();
//         })
//         .then(data => {
//             console.log("Fetched data:", data);
//             const tableBody = document.getElementById("teamsTableBody");
//             tableBody.innerHTML = ""; // 清空表格的舊數據

//             if (!data || !Array.isArray(data) || data.length === 0) {
//                 // 如果數據是空的或有錯誤訊息，顯示一條消息
//                 tableBody.innerHTML = `<tr><td colspan="9" class="text-center">沒有可用的數據</td></tr>`;
//                 return;
//             }

//             // 插入新的數據行
//             data.forEach((team, index) => {
//                 const prScore = team["PR分數"] !== undefined && team["PR分數"] !== null ? team["PR分數"].toFixed(2) : "N/A";
//                 const officialRank = team["official_rank"] !== null && team["official_rank"] !== undefined ? team["official_rank"] : "無";
//                 const teamName = team["隊伍"] ? team["隊伍"] : "未知隊伍";
//                 const region = team["visitor_region"] || team["home_region"] || "未知地區";
//                 const wins = team["勝場"] !== undefined && team["勝場"] !== null ? team["勝場"] : "無";
//                 const losses = team["敗場"] !== undefined && team["敗場"] !== null ? team["敗場"] : "無";

//                 console.log(`Processing team: ${teamName}, PR分數: ${prScore}, 勝場: ${wins}, 敗場: ${losses}`);

//                 const row = document.createElement('tr');
//                 row.innerHTML = `
//                     <td>${index + 1}</td>
//                     <td>${selectedYear}</td>
//                     <td>${teamName}</td>
//                     <td>${region}</td>
//                     <td>${prScore}</td>
//                     <td>${officialRank}</td>
//                     <td>${wins}</td>
//                     <td>${losses}</td>
//                     <td>備註 - 無</td>
//                 `;
//                 tableBody.appendChild(row);
//             });
//         })
//         .catch(error => {
//             console.error("錯誤:", error);
//             alert("發生錯誤，無法載入資料: " + error.message);
//         });
// }
