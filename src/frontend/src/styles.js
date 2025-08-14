import { makeStyles } from '@mui/styles';

export const useStyles = makeStyles((theme) => {
  const spacing = typeof theme.spacing === 'function' ? theme.spacing : (factor) => `${0.25 * factor}rem`;

  return {
    summarySection: {
      marginBottom: spacing(2),
    },
    sectionTitle: {
      fontWeight: 'bold',
      marginBottom: spacing(1),
    },
    bulletList: {
      paddingLeft: spacing(2),
      marginTop: 0,
      marginBottom: spacing(1),
    },
    bulletItem: {
      display: 'flex',
      alignItems: 'flex-start',
      marginBottom: spacing(0.5),
    },
    bulletIcon: {
      minWidth: '24px',
      marginTop: '4px',
      marginRight: spacing(1),
    },
    bulletText: {
      margin: 0,
    },
    paragraph: {
      marginBottom: spacing(1),
    },
  };
});