-- Task 1: Fix Sundong Kim's profile image (.jpeg → .png)
UPDATE users SET profile_image = 'https://sundong.kim/assets/img/members/sundongkim.png' WHERE user_id = 'sundong';

-- Task 2: Remove iamseungpil (admin) submission records
DELETE FROM submissions WHERE user_id = 'iamseungpil';
DELETE FROM evaluations WHERE user_id = 'iamseungpil';
DELETE FROM personal_timers WHERE user_id = 'iamseungpil';

-- Task 3: Recalculate time ranks for all remaining evaluations
-- Rank bonus: 1st=20, 2nd=17, 3rd=14, 4th=11, 5th=8, 6th+=5

-- Week 1: sundong(82.4) → rank1, azamat1(158.4) → rank2, jhjh6612(4679.2) → rank3
UPDATE evaluations SET time_rank = 1, time_rank_bonus = 20, total_score = rubric_score + 20 WHERE user_id = 'sundong' AND week = 1;
UPDATE evaluations SET time_rank = 2, time_rank_bonus = 17, total_score = rubric_score + 17 WHERE user_id = 'azamat1' AND week = 1;
UPDATE evaluations SET time_rank = 3, time_rank_bonus = 14, total_score = rubric_score + 14 WHERE user_id = 'jhjh6612' AND week = 1;

-- Week 2: omnyx2(9.3) → rank1, sweetautumnfox(21.8) → rank2, whatchang(32.2) → rank3,
--          BATMANbf1(47.5) → rank4, koodol(54.1) → rank5, jhjh6612(232.8) → rank6
--          sundong has no week2 submission → rank0
UPDATE evaluations SET time_rank = 0, time_rank_bonus = 0, total_score = rubric_score + 0 WHERE user_id = 'sundong' AND week = 2;
UPDATE evaluations SET time_rank = 1, time_rank_bonus = 20, total_score = rubric_score + 20 WHERE user_id = 'omnyx2' AND week = 2;
UPDATE evaluations SET time_rank = 2, time_rank_bonus = 17, total_score = rubric_score + 17 WHERE user_id = 'sweetautumnfox' AND week = 2;
UPDATE evaluations SET time_rank = 3, time_rank_bonus = 14, total_score = rubric_score + 14 WHERE user_id = 'whatchang' AND week = 2;
UPDATE evaluations SET time_rank = 4, time_rank_bonus = 11, total_score = rubric_score + 11 WHERE user_id = 'BATMANbf1' AND week = 2;
UPDATE evaluations SET time_rank = 5, time_rank_bonus = 8, total_score = rubric_score + 8 WHERE user_id = 'koodol' AND week = 2;
UPDATE evaluations SET time_rank = 6, time_rank_bonus = 5, total_score = rubric_score + 5 WHERE user_id = 'jhjh6612' AND week = 2;
