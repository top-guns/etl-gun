export const getError = async (call: () => void): Promise<any> => {
    try {
      await call();
    } catch (error) {
      return error;
    }
};