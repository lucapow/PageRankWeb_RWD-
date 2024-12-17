USE sports_data;

ALTER TABLE match_data
ADD CONSTRAINT fk_visitor
FOREIGN KEY (Visitor_id) REFERENCES highschool(HighSchool_id);

ALTER TABLE match_data
ADD CONSTRAINT fk_home
FOREIGN KEY (Home_id) REFERENCES highschool(HighSchool_id);
CREATE VIEW County_Scores AS
SELECT h.County,
       SUM(CASE WHEN m.Visitor_id = h.HighSchool_id THEN m.PTS_Visit ELSE 0 END) AS Total_Visitor_Points,
       SUM(CASE WHEN m.Home_id = h.HighSchool_id THEN m.PTS_Home ELSE 0 END) AS Total_Home_Points,
       SUM(CASE WHEN m.Visitor_id = h.HighSchool_id THEN m.PTS_Visit ELSE 0 END +
           CASE WHEN m.Home_id = h.HighSchool_id THEN m.PTS_Home ELSE 0 END) AS Total_County_Points
FROM match_data m
LEFT JOIN highschool h ON m.Visitor_id = h.HighSchool_id OR m.Home_id = h.HighSchool_id
GROUP BY h.County;
SELECT * FROM County_Scores;