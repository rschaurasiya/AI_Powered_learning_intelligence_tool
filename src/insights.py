"""
Insights Module
Calculate student risk scores and chapter difficulty metrics.
"""

import pandas as pd
import numpy as np


class InsightsGenerator:
    """Generate learning analytics insights from student data."""
    
    def __init__(self):
        """Initialize InsightsGenerator."""
        pass
    
    def calculate_risk_score(self, student_data):
        """
        Calculate risk score for a student or group of students.
        
        Risk levels are determined by:
        - HIGH: completion_rate < 0.3 AND avg_time < 2 hours
        - MEDIUM: completion_rate < 0.6
        - LOW: otherwise
        
        Args:
            student_data (pd.DataFrame): Student data with Time_Spent_Hours and Completed columns
            
        Returns:
            str or pd.Series: Risk flag ("HIGH", "MEDIUM", or "LOW")
        """
        # Calculate metrics
        avg_time = student_data['Time_Spent_Hours'].mean()
        completion_rate = student_data['Completed'].mean()
        
        # Determine risk level based on specified criteria
        if completion_rate < 0.3 and avg_time < 2:
            risk_flag = "HIGH"
        elif completion_rate < 0.6:
            risk_flag = "MEDIUM"
        else:
            risk_flag = "LOW"
        
        return risk_flag
    
    def calculate_student_risk_scores(self, df):
        """
        Calculate risk scores for each student individually.
        
        Args:
            df (pd.DataFrame): Full dataset with Student_ID, Time_Spent_Hours, and Completed
            
        Returns:
            pd.DataFrame: Student IDs with their risk scores
        """
        student_risks = []
        
        for student_id in df['Student_ID'].unique():
            student_data = df[df['Student_ID'] == student_id]
            risk_score = self.calculate_risk_score(student_data)
            
            student_risks.append({
                'Student_ID': student_id,
                'Risk_Score': risk_score,
                'Avg_Time_Spent': student_data['Time_Spent_Hours'].mean(),
                'Completion_Rate': student_data['Completed'].mean(),
                'Total_Chapters': len(student_data)
            })
        
        return pd.DataFrame(student_risks)
    
    def calculate_chapter_difficulty(self, df):
        """
        Calculate difficulty score for each chapter.
        
        Difficulty is calculated as:
        - Normalize time: time_normalized = time_spent / max_time
        - Difficulty = (1 - completion_rate) * 0.6 + time_normalized * 0.4
        
        Args:
            df (pd.DataFrame): Dataset with Chapter_Order, Completed, and Time_Spent_Hours
            
        Returns:
            pd.DataFrame: Chapter statistics with difficulty scores
        """
        # Group by chapter and calculate statistics
        chapter_stats = df.groupby('Chapter_Order').agg({
            'Completed': 'mean',              # % who completed
            'Time_Spent_Hours': 'mean'        # Avg time spent
        }).reset_index()
        
        # Rename columns for clarity
        chapter_stats.columns = ['Chapter_Order', 'Completion_Rate', 'Avg_Time_Spent']
        
        # Normalize time (0-1 scale)
        max_time = chapter_stats['Avg_Time_Spent'].max()
        if max_time > 0:
            chapter_stats['Time_Normalized'] = chapter_stats['Avg_Time_Spent'] / max_time
        else:
            chapter_stats['Time_Normalized'] = 0
        
        # Calculate difficulty score
        chapter_stats['Difficulty_Score'] = (
            (1 - chapter_stats['Completion_Rate']) * 0.6 +     # Low completion = harder
            chapter_stats['Time_Normalized'] * 0.4             # High time = harder
        )
        
        # Sort by difficulty (hardest first)
        chapter_stats = chapter_stats.sort_values('Difficulty_Score', ascending=False)
        
        return chapter_stats
    
    def get_chapter_insights(self, df):
        """
        Get comprehensive chapter insights including difficulty and engagement.
        
        Args:
            df (pd.DataFrame): Dataset with chapter information
            
        Returns:
            pd.DataFrame: Detailed chapter insights
        """
        chapter_stats = self.calculate_chapter_difficulty(df)
        
        # Add additional insights
        chapter_details = df.groupby('Chapter_Order').agg({
            'Completed': ['mean', 'count'],
            'Time_Spent_Hours': ['mean', 'std', 'min', 'max'],
            'Scores': 'mean'
        }).reset_index()
        
        # Flatten column names
        chapter_details.columns = [
            'Chapter_Order', 
            'Completion_Rate', 'Total_Students',
            'Avg_Time', 'Time_Std', 'Min_Time', 'Max_Time',
            'Avg_Score'
        ]
        
        # Merge with difficulty scores
        chapter_insights = chapter_stats.merge(
            chapter_details[['Chapter_Order', 'Total_Students', 'Time_Std', 'Avg_Score']], 
            on='Chapter_Order'
        )
        
        # Add difficulty category
        chapter_insights['Difficulty_Category'] = pd.cut(
            chapter_insights['Difficulty_Score'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Easy', 'Medium', 'Hard']
        )
        
        return chapter_insights
    
    def generate_summary_report(self, df):
        """
        Generate a comprehensive summary report of all insights.
        
        Args:
            df (pd.DataFrame): Full dataset
            
        Returns:
            dict: Summary report with all insights
        """
        report = {}
        
        # Overall statistics
        report['overall'] = {
            'total_students': df['Student_ID'].nunique(),
            'total_courses': df['Course_ID'].nunique(),
            'total_chapters': df['Chapter_Order'].nunique(),
            'overall_completion_rate': df['Completed'].mean(),
            'avg_time_per_chapter': df['Time_Spent_Hours'].mean(),
            'avg_score': df['Scores'].mean()
        }
        
        # Student risk distribution
        student_risks = self.calculate_student_risk_scores(df)
        report['risk_distribution'] = {
            'high_risk': (student_risks['Risk_Score'] == 'HIGH').sum(),
            'medium_risk': (student_risks['Risk_Score'] == 'MEDIUM').sum(),
            'low_risk': (student_risks['Risk_Score'] == 'LOW').sum()
        }
        
        # Chapter difficulty
        chapter_difficulty = self.calculate_chapter_difficulty(df)
        report['chapter_insights'] = {
            'hardest_chapter': int(chapter_difficulty.iloc[0]['Chapter_Order']),
            'easiest_chapter': int(chapter_difficulty.iloc[-1]['Chapter_Order']),
            'avg_difficulty': float(chapter_difficulty['Difficulty_Score'].mean())
        }
        
        # Store detailed dataframes
        report['student_risk_details'] = student_risks
        report['chapter_difficulty_details'] = chapter_difficulty
        
        return report


# Convenience functions
def calculate_risk_score(student_data):
    """
    Calculate risk score for student data.
    
    Args:
        student_data (pd.DataFrame): Student data
        
    Returns:
        str: Risk score ("HIGH", "MEDIUM", or "LOW")
    """
    generator = InsightsGenerator()
    return generator.calculate_risk_score(student_data)


def calculate_chapter_difficulty(df):
    """
    Calculate chapter difficulty scores.
    
    Args:
        df (pd.DataFrame): Dataset with chapter information
        
    Returns:
        pd.DataFrame: Chapter difficulty statistics
    """
    generator = InsightsGenerator()
    return generator.calculate_chapter_difficulty(df)


def generate_summary_report(df):
    """
    Generate comprehensive insights report.
    
    Args:
        df (pd.DataFrame): Full dataset
        
    Returns:
        dict: Summary report
    """
    generator = InsightsGenerator()
    return generator.generate_summary_report(df)
